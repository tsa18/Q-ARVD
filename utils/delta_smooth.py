import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Literal, Optional
from quant.quant_layer import QuantModule
import logging
import copy
from quant.quant_layer import UniformAffineQuantizer
from typing import Literal, Optional
logger = logging.getLogger(__name__)



# def find_outlier_channels(
#     model: nn.Module,
#     top_percent: float = 5.0,
#     layer_name_filter: Optional[str] = None,
# ) -> nn.Module:
    
#     # 遍历模型所有层
#     for name, module in model.named_modules():
#         # 只处理线性层 + 名称过滤
#         if not isinstance(module, QuantModule):
#             continue
#         if layer_name_filter and all(filter not in name for filter in layer_name_filter):
#             # print('skip:', name)
#             continue
#         # ============== 核心：全程在原设备（GPU/CPU）上运算，无数据拷贝 ==============
#         device = module.weight.device
#         dtype = module.weight.dtype
#         # 分离权重（不计算梯度），转为浮点计算
#         W = module.weight.detach().to(torch.float32)  # shape: [out_features, in_features]
#         # 1. 计算输入通道的 L2 范数 (dim=0 按列计算)
#         col_norms = W.norm(dim=0)  # shape: [in_features]
#         n_channels = col_norms.shape[0]
#         # n_outlier = max(1, int(np.ceil(n_channels * top_percent / 100.0)))
#         n_outlier = (max(1, int(np.ceil(n_channels * top_percent / 100.0))) + 31) // 32 * 32
#         # 2. 排序：筛选异常通道（范数最大的top%）和正常通道
#         sorted_indices = torch.argsort(col_norms)
#         outlier_indices = sorted_indices[-n_outlier:]  # 异常通道索引
#         normal_indices = sorted_indices[:-n_outlier]  # 正常通道索引
#         ## 3. 设置层
#         module.outlier_indices = outlier_indices
#         module.normal_indices = normal_indices
#         module.use_dual_scale = True
#         module.W_outlier = module.weight[:, outlier_indices].to(dtype=dtype, device=device).contiguous()
#         module.W_normal = module.weight[:, normal_indices].to(dtype=dtype, device=device).contiguous()
#         module.weight = None

#        ## 4. 设置双量化器（权重 + 激活 均拆分为独立2个）
#         # ===================== 权重量化器（原有逻辑，保留） =====================
#         original_weight_params = module.weight_quant_params
#         original_weight_name = module.weight_quantizer.name
#         module.weight_quantizer_outlier = UniformAffineQuantizer(
#             **original_weight_params,
#             name=f"{original_weight_name}_outlier",
#             qtype='weight'
#         )
#         module.weight_quantizer_normal = UniformAffineQuantizer(
#             **original_weight_params,
#             name=f"{original_weight_name}_normal",
#             qtype='weight'
#         )
#         # ===================== 新增：激活量化器（完全对称拆分） =====================
#         original_act_params = module.act_quant_params  # 原激活量化参数
#         original_act_name = module.act_quantizer.name  # 原激活量化器名称
#         module.act_quantizer_outlier = UniformAffineQuantizer(
#             **original_act_params,
#             name=f"{original_act_name}_outlier",
#             qtype='act'
#         )
#         module.act_quantizer_normal = UniformAffineQuantizer(
#             **original_act_params,
#             name=f"{original_act_name}_normal",
#             qtype='act'
#         )
#         # 清空原有单量化器，防止冲突
#         module.weight_quantizer = None
#         module.act_quantizer = None
#         logger.info(f"process outliers of {module.name}, top {top_percent}%, n_outlier={n_outlier}")




# ## IQR：1. 对于平滑段，加入最小阈值。2. 对于快速下降，用后 75% 做 IQR，放松outlier 判断
# def detect_outlier_count(
#     col_norms: torch.Tensor,
#     method: Literal["iqr", "mad_zscore"] = "iqr",
#     align: int = 32,
#     iqr_k: float = 1.5,
#     mad_threshold: float = 3.5,
#     min_ratio_vs_median: float = 1.2,
#     ref_upper_pct: float = 75.0,
# ) -> tuple[int, float]:                              # ← 返回类型改为 tuple
    
#     norms_np = col_norms.float().cpu().numpy()
#     n = len(norms_np)
#     median = float(np.median(norms_np))
#     if method == "iqr":
#         print(f'ref_upper_pct={ref_upper_pct}')
#         ref_boundary = float(np.percentile(norms_np, ref_upper_pct))
#         ref_vals = norms_np[norms_np <= ref_boundary]
#         q1, q3 = np.percentile(ref_vals, [25, 75])
#         iqr = q3 - q1
#         stat_threshold = q3 + iqr_k * iqr
#     elif method == "mad_zscore":
#         mad = float(np.median(np.abs(norms_np - median)))
#         if mad < 1e-8:
#             return 0, float(median)              # ← 同步返回 tuple
#         stat_threshold = median + (mad_threshold / 0.6745) * mad
#     else:
#         raise ValueError(f"Unknown method: {method}")
#     print(f"min_ratio_vs_median={min_ratio_vs_median}")
#     effective_threshold = max(stat_threshold, min_ratio_vs_median * median)
#     n_outlier = int(np.sum(norms_np > effective_threshold))

#     if n_outlier > 0 and align > 1:
#         n_outlier = (n_outlier + align - 1) // align * align
#     n_outlier = min(n_outlier, n // 2)
#     return n_outlier, float(effective_threshold)   # ← 同时返回 threshold


def _plot_layer_outliers(
    ax: plt.Axes,
    name: str,
    norms_np: np.ndarray,
    outlier_indices: np.ndarray,
    threshold: Optional[float] = None,            # ← 新增参数，直接接收外部传入
):
    C_NORMAL  = "#aec7e8"
    C_OUTLIER = "#d62728"
    C_MEAN    = "steelblue"
    C_THRESH  = "#ff7f0e"

    n_ch      = len(norms_np)
    n_outlier = len(outlier_indices)
    short     = ".".join(name.split(".")[-2:]) if "." in name else name

    sorted_order  = np.argsort(norms_np)[::-1]
    sorted_norms  = norms_np[sorted_order]
    x             = np.arange(n_ch)

    outlier_set   = set(outlier_indices.tolist())
    is_outlier    = np.array([sorted_order[i] in outlier_set for i in range(n_ch)])

    ax.plot(x, sorted_norms, linewidth=0.8, color=C_NORMAL, zorder=2)
    ax.fill_between(x, sorted_norms, alpha=0.20, color=C_NORMAL, zorder=1)

    if n_outlier > 0:
        outlier_x     = x[is_outlier]
        outlier_norms = sorted_norms[is_outlier]
        ax.plot(outlier_x, outlier_norms, linewidth=0.8, color=C_OUTLIER, zorder=3,
                label=f"outlier  n={n_outlier} ({100 * n_outlier / n_ch:.1f}%)")
        ax.fill_between(outlier_x, outlier_norms, alpha=0.20, color=C_OUTLIER, zorder=1)
        ax.axvspan(0, n_outlier - 0.5, alpha=0.06, color=C_OUTLIER, zorder=0)

    mean_val = sorted_norms.mean()
    ax.axhline(mean_val, color=C_MEAN, linewidth=0.9, linestyle="--",
               label=f"mean={mean_val:.3f}", zorder=3)

    # ── 阈值线：直接使用传入的 threshold，不再自行计算 ────────
    if threshold is not None:
        ax.axhline(threshold, color=C_THRESH, linewidth=0.9,
                   linestyle=":", label=f"threshold={threshold:.3f}", zorder=3)

    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_xlabel("In-Channel (sorted by norm ↓)", fontsize=8)
    ax.set_ylabel("L2 Norm", fontsize=8)
    ax.set_title(short, fontsize=9, pad=4)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-n_ch * 0.02, n_ch)
    ax.set_ylim(bottom=0)








# ────────────────────────────────────────────────
#  核心：自动检测异常通道数
# ────────────────────────────────────────────────
# 原始 IQR+MAD
def detect_outlier_count(
    col_norms: torch.Tensor,
    method: Literal["iqr", "mad_zscore"] = "iqr",
    align: int = 32,
    iqr_k: float = 1.5,
    mad_threshold: float = 3.5,
    min_ratio_vs_median: float = 1.2,
) -> tuple[int, float]:
    """
    根据列范数自动确定异常通道的数量。

    Args:
        col_norms  : 每个输入通道的 L2 范数，shape [in_features]
        method     : 检测方法
        align      : 对齐到该值的倍数（硬件友好，设为 1 可禁用）
        iqr_k      : IQR 方法的倍数系数（1.5=标准离群，3.0=极端离群）
        mad_threshold : MAD Z-score 的阈值（推荐 3.5）
    Returns:
        n_outlier  : 异常通道数（已 align 对齐，最小为 0）
    """
    norms_np = col_norms.float().cpu().numpy()
    n = len(norms_np)

    # print(f"min_ratio_vs_median={min_ratio_vs_median}")
    if method == "iqr":
        q1, q3 = np.percentile(norms_np, [25, 75])
        iqr = q3 - q1
        threshold = q3 + iqr_k * iqr
        median = float(np.median(norms_np))
        threshold = max(threshold, min_ratio_vs_median * median)
        n_outlier = int(np.sum(norms_np > threshold))

    elif method == "mad_zscore":
        median = float(np.median(norms_np))
        mad = np.median(np.abs(norms_np - median))
        if mad < 1e-8:          # 所有值几乎相同，没有离群
            return 0, 0.0
        modified_z = 0.6745 * (norms_np - median) / mad
        # n_outlier = int(np.sum(modified_z > mad_threshold))
        n_outlier = int(np.sum(
            (modified_z > mad_threshold) & (norms_np > min_ratio_vs_median * median)
        ))

        mad_based_threshold = median + (mad_threshold * mad) / 0.6745
        threshold = max(mad_based_threshold, min_ratio_vs_median * median)

    else:
        raise ValueError(f"Unknown method: {method}")

    # 对齐到 align 的整数倍（0 不对齐，保留为 0）
    if n_outlier > 0 and align > 1:
        n_outlier = (n_outlier + align - 1) // align * align

    # 防止异常：不能超过通道总数的 50%
    n_outlier = min(n_outlier, n // 2)
    return n_outlier, threshold



def find_outlier_channels(
    model: nn.Module,
    method: Literal["iqr", "mad_zscore"] = "iqr",
    layer_name_filter: Optional[list[str]] = None,
    align: int = 32,
    iqr_k: float = 1.5,
    mad_threshold: float = 3.5,
    min_ratio_vs_median: float = 1.2,
    # ── 可视化参数 ──────────────────────────────
    visualize: bool = False,
    vis_cols: int = 3,
    figsize_per_cell: tuple = (4.0, 3.5),
    save_path: Optional[str] = None,
    show: bool = True,
    only_mark = False
) -> nn.Module:
    detect_kwargs = dict(
        align=align, iqr_k=iqr_k,
        mad_threshold=mad_threshold, 
        min_ratio_vs_median=min_ratio_vs_median,
    )

    # visualize=True 时，先收集绘图数据，最后统一出图
    vis_records: list[dict] = []   # [{"name", "norms_np", "outlier_indices"}, ...]
    all_layers = []
    applied_layers = []

    for name, module in model.named_modules():
        if not isinstance(module, QuantModule):
            continue
        if layer_name_filter and all(f not in name for f in layer_name_filter):
            continue
        all_layers.append(name)
        device = module.weight.device
        dtype  = module.weight.dtype
        W      = module.weight.detach().to(torch.float32)
        col_norms = W.norm(dim=0)

     
        n_outlier, threshold = detect_outlier_count(col_norms, method=method, **detect_kwargs)

        if only_mark:
            if n_outlier > 0:
                module.is_out_layer = True
                module.act_quantizer.is_out_layer = True
            else:
                module.is_out_layer = False
                module.act_quantizer.is_out_layer = False
            continue

        if n_outlier == 0:
            logger.info(f"{name}: no outliers detected, skip dual-scale")
            if visualize:
                vis_records.append({
                    "name": name,
                    "norms_np": col_norms.cpu().numpy(),
                    "outlier_indices": np.array([], dtype=int),
                    "threshold": threshold,   
                })
            continue
        applied_layers.append(name)

        sorted_indices  = torch.argsort(col_norms)
        outlier_indices = sorted_indices[-n_outlier:]
        normal_indices  = sorted_indices[:-n_outlier]

        # ── 收集绘图数据（在权重被拆分前） ──────────
        if visualize:
            vis_records.append({
                "name": name,
                "norms_np": col_norms.cpu().numpy(),
                "outlier_indices": outlier_indices.cpu().numpy(),
                "threshold": threshold,   
            })

        # ── 权重拆分 ─────────────────────────────────
        module.outlier_indices = outlier_indices
        module.normal_indices  = normal_indices
        module.use_dual_scale  = True
        module.W_outlier = module.weight[:, outlier_indices].to(dtype=dtype, device=device).contiguous()
        module.W_normal  = module.weight[:, normal_indices ].to(dtype=dtype, device=device).contiguous()
        module.weight    = None

        # ── 量化器拆分 ───────────────────────────────
        for qtype, attr_orig in [("weight", "weight_quantizer"), ("act", "act_quantizer")]:
            orig_q      = getattr(module, attr_orig)
            orig_params = getattr(module, f"{qtype}_quant_params")
            setattr(module, f"{attr_orig}_outlier", UniformAffineQuantizer(
                **orig_params, name=f"{orig_q.name}_outlier", qtype=qtype, use_dual_scale=True))
            setattr(module, f"{attr_orig}_normal", UniformAffineQuantizer(
                **orig_params, name=f"{orig_q.name}_normal", qtype=qtype,use_dual_scale=True))
            setattr(module, attr_orig, None)

        logger.info(
            f"{name}: method={method}, n_outlier={n_outlier}/{col_norms.shape[0]} "
            f"({100 * n_outlier / col_norms.shape[0]:.1f}%)"
        )

    # ── 统一出图 ──────────────────────────────────────
    if visualize and vis_records:
        n      = len(vis_records)
        rows   = (n + vis_cols - 1) // vis_cols
        fig, axes = plt.subplots(
            rows, vis_cols,
            figsize=(figsize_per_cell[0] * vis_cols, figsize_per_cell[1] * rows),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(-1)

        for idx, rec in enumerate(vis_records):
            _plot_layer_outliers(
                ax=axes[idx],
                name=rec["name"],
                norms_np=rec["norms_np"],
                outlier_indices=rec["outlier_indices"],
                threshold=rec["threshold"],   
            )
        for idx in range(n, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"Per-Channel L2 Norm & Outlier Detection  [method={method}]",
            fontsize=11,
        )
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        if show:
            plt.show()

    layer_types = ['self_attn.attn.q', 'self_attn.attn.k', 'self_attn.attn.v', 'self_attn.attn.o',
                  'cross_attn.cross_attn.q','cross_attn.cross_attn.k','cross_attn.cross_attn.v','cross_attn.cross_attn.o',
                  'ffn.0', 'ffn.2']
    for layer_type in layer_types:
        total_num = sum([layer_type in l for l in all_layers])
        applied_num = sum([layer_type in l for l in applied_layers])
        logger.info(f"{layer_type}: applied ratio={applied_num}/{total_num}={applied_num/total_num*100:.2f}%")

    logger.info(f"{len(applied_layers)}/{len(all_layers)}={len(applied_layers)/len(all_layers)*100:.2f}% layers apply dual scale!")

    

    return model