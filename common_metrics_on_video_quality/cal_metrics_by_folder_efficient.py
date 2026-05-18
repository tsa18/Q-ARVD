import torch
import os
import glob
import json
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from calculate_lpips import calculate_lpips
from calculate_fvd import trans

torch.set_grad_enabled(False)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--gen_videos_dir", type=str)
parser.add_argument("--gt_videos_dir", type=str)
args = parser.parse_args()

# ===================== 极致优化配置 =====================
NUM_WORKERS = 4          # 视频读取线程数
BATCH_SIZE = 64           # 🌟 新增：批处理大小！如果显存/内存还爆，就把这个调小(如 4 或 2)
# ========================================================

from decord import VideoReader, cpu
def load_video_fast(path: str) -> torch.Tensor:
    vr = VideoReader(path, ctx=cpu(0), num_threads=4)
    video = vr.get_batch(range(len(vr))).asnumpy()
    video = torch.from_numpy(video).float() / 255.0
    video = video.permute(0, 3, 1, 2)
    del vr
    return video

def load_pair_global(idx, gt_paths, gen_paths):
    gt = load_video_fast(gt_paths[idx])
    gen = load_video_fast(gen_paths[idx])
    return gt, gen

def calculate_video_metrics(
    gt_folder: str,
    gen_folder: str,
    device: str | torch.device = "cuda",
    fvd_method: str = "styleganv",
    only_final: bool = True,
    video_suffix: tuple = (".mp4", ".avi", ".mov", ".mkv")
):
    if not os.path.isdir(gt_folder): raise ValueError(f"真实视频文件夹不存在: {gt_folder}")
    if not os.path.isdir(gen_folder): raise ValueError(f"生成视频文件夹不存在: {gen_folder}")

    def get_videos(folder):
        files = glob.glob(os.path.join(folder, "*"))
        return sorted([f for f in files if f.lower().endswith(video_suffix)])
    
    gt_paths = get_videos(gt_folder)
    gen_paths = get_videos(gen_folder)
    
    assert len(gt_paths) == len(gen_paths), "视频数量不匹配"
    for g, ge in zip(gt_paths, gen_paths):
        assert os.path.basename(g) == os.path.basename(ge), "文件名不匹配"
    total = len(gt_paths)
    print(f"✅ 找到 {total} 组视频，使用 BATCH_SIZE={BATCH_SIZE} 分批处理，告别内存爆炸！")

    device = torch.device(device)

    # 🌟 动态导入 FVD 的底层特征提取函数
    if fvd_method == 'styleganv':
        from fvd.styleganv.fvd import get_fvd_feats, frechet_distance, load_i3d_pretrained
    elif fvd_method == 'videogpt':
        from fvd.videogpt.fvd import load_i3d_pretrained, frechet_distance
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats

    # 提前加载 I3D 模型（只需加载一次）
    print("⏳ 正在加载 I3D 模型用于 FVD 计算...")
    i3d = load_i3d_pretrained(device=device)

    # 用于存放所有批次的特征和指标
    all_gt_fvd_feats = []
    all_gen_fvd_feats = []
    lpips_list = []

    # ===================== 核心：分批加载和计算 =====================
    for i in tqdm(range(0, total, BATCH_SIZE), desc="📦 分批处理进度"):
        batch_gt_paths = gt_paths[i : i + BATCH_SIZE]
        batch_gen_paths = gen_paths[i : i + BATCH_SIZE]
        current_batch_size = len(batch_gt_paths)

        # 1. 多线程加载当前批次的视频
        batch_pairs = [None] * current_batch_size
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = {executor.submit(load_pair_global, j, batch_gt_paths, batch_gen_paths): j for j in range(current_batch_size)}
            for future in as_completed(futures):
                idx = futures[future]
                batch_pairs[idx] = future.result()

        # 2. 组装当前批次的 Tensor
        gt_batch = torch.stack([p[0] for p in batch_pairs])
        gen_batch = torch.stack([p[1] for p in batch_pairs])
        del batch_pairs # 释放临时列表

        with torch.no_grad():
            # -------- 3. 提取当前批次的 FVD 特征 --------
            # trans(): 转换形状 BTCHW -> BCTHW (由 calculate_fvd.py 提供)
            gt_trans = trans(gt_batch)
            gen_trans = trans(gen_batch)

            # 提取特征
            feat_gt = get_fvd_feats(gt_trans, i3d=i3d, device=device)
            feat_gen = get_fvd_feats(gen_trans, i3d=i3d, device=device)
            
            all_gt_fvd_feats.append(feat_gt)
            all_gen_fvd_feats.append(feat_gen)

            # -------- 4. 计算当前批次的 LPIPS --------
            # 为了防止显存峰值，LPIPS 还是逐个计算
            for j in range(current_batch_size):
                gt_vid = gt_batch[j].unsqueeze(0)
                gen_vid = gen_batch[j].unsqueeze(0)
                lpips_res = calculate_lpips(gt_vid, gen_vid, device, only_final=True)
                lpips_list.append(lpips_res['value'])

        # 5. 🧹 清理内存和显存（关键步骤）
        del gt_batch, gen_batch, gt_trans, gen_trans, feat_gt, feat_gen
        torch.cuda.empty_cache() 

    # ===================== 所有批次处理完毕，统计算终值 =====================
    print("\n📊 正在计算最终分布距离 (FVD)...")
    # 将所有批次的特征拼接在一起
    all_gt_fvd_feats = np.concatenate(all_gt_fvd_feats, axis=0)
    all_gen_fvd_feats = np.concatenate(all_gen_fvd_feats, axis=0)

    # 计算 Fréchet Distance
    final_fvd = frechet_distance(all_gt_fvd_feats, all_gen_fvd_feats)

    result = {
        'fvd': float(final_fvd),
        'lpips': float(np.mean(lpips_list))
    }

    print(f"✅ 所有指标计算完成！")
    return result

# ------------------- 调用示例 -------------------
if __name__ == "__main__":
    
    GT_FOLDER = args.gt_videos_dir
    GEN_FOLDER = args.gen_videos_dir
    print(GT_FOLDER, GEN_FOLDER)

    metrics = calculate_video_metrics(
        gt_folder=GT_FOLDER,
        gen_folder=GEN_FOLDER,
        device="cuda",
        fvd_method="styleganv",
        only_final=True
    )

    print("\n" + "="*50)
    print("📊 视频指标计算结果：")
    print("="*50)
    print(json.dumps(metrics, indent=4))