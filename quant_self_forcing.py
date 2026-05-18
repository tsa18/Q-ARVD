# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import pipeline
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import logging
import argparse, os
from quant.quant_model_self_forcing import QuantModelSelfForcing
import torch.distributed as dist
from utils.misc import set_seed
from omegaconf import OmegaConf
from tqdm import tqdm
from torchvision.io import write_video
from einops import rearrange
from omegaconf import OmegaConf
from pipeline import (
    CausalInferencePipeline,
)
from quant.layer_recon_self_forcing import reconstruct
from quant.quant_layer import UniformAffineQuantizer
from snapshot import Snapshot
import sys
from utils.outlier_adaptive_dual_scale import find_outlier_channels

gpu = torch.device(f'cuda:{torch.cuda.current_device()}')
if "LOCAL_RANK" in os.environ:
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size()
    set_seed(42 + local_rank)
else:
    device = torch.device("cuda")
    local_rank = 0
    world_size = 1
    set_seed(42)

logger = logging.getLogger(__name__)

def main(args):

    # Setup save path:
    exp_path = 'exp_logs'
    os.makedirs(exp_path, exist_ok=True)
    exp_save_path = os.path.join(exp_path, f"{args.exp_name}-w{args.weight_bit}a{args.act_bit}")
    os.makedirs(exp_save_path, exist_ok=True)
    log_path = os.path.join(exp_save_path, "run.log")
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Arguments: {args}")
    logger.info(f"Saving to {exp_save_path}")

    # save exp snapshot
    snap = Snapshot(
        command="python " + " ".join(sys.argv),
        save_dir=exp_save_path,
    )
    snap.save()

    # Setup PyTorch:
    torch.manual_seed(args.seed)
    torch.set_grad_enabled(False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    #### 1. Load model
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    # initialize pipeline
    assert hasattr(config, 'denoising_step_list')
    # few-step inference
    pipeline = CausalInferencePipeline(config, device=device)
    if args.checkpoint_path:
        state_dict = torch.load(args.checkpoint_path, map_location="cpu")
        pipeline.generator.load_state_dict(state_dict['generator' if not args.use_ema else 'generator_ema'])
    
    pipeline = pipeline.to(dtype=torch.bfloat16)
    pipeline.text_encoder.to(device=gpu)
    pipeline.generator.to(device=gpu)
    pipeline.vae.to(device=gpu)
    logger.info(f"Loaded {pipeline.generator.model} with {sum(p.numel() for p in pipeline.generator.model.parameters())/(10**9):,} B parameters.")

    #### 2. Setup quantization:
    a_scale_method = 'mse'
    # a_scale_method = 'max'
    wq_params = {'n_bits': args.weight_bit, 'channel_wise': True, 'scale_method': 'mse'}
    aq_params = {'n_bits': args.act_bit, 'symmetric': False, 'channel_wise': False, 'scale_method': a_scale_method, 'leaf_param': True}
    #### 3. Build quantized model
    pipeline.generator.model = QuantModelSelfForcing(
            model=pipeline.generator.model, weight_quant_params=wq_params, act_quant_params=aq_params)
    pipeline.eval()
    logger.info(pipeline.generator.model)
    # set sensitivity 
    sensitivity = [float(x) for x in args.sensitivity.split(',')]
    pipeline.generator.model.set_sensitivity(sensitivity)
    pipeline.generator.model.set_quant_state(False, False)
    print(pipeline.generator.model)
    logger.info('================== Finish quant model building ==================')

    # find outliers and use dual-scale quantization
    find_outlier_channels(
        pipeline.generator.model,
        method='mad_zscore',
        mad_threshold=3.5,
        min_ratio_vs_median = 1.2,
        layer_name_filter=['block'],
    )

    #### 4. Inference with quantized model
    # Create dataset
    with open(args.data_path, 'r') as f:
        lines = f.readlines()
        all_prompts = [line.strip() for line in lines]

    act_init_total_round = 1
    collected_recon_data_round = 1
    all_prompts = [all_prompts[0]] + all_prompts[0:act_init_total_round] \
                  + all_prompts[0:collected_recon_data_round] + all_prompts

    for i, data in tqdm(enumerate(all_prompts), disable=(local_rank != 0)):
        logger.info(f'-------------------- i={i} --------------------')
        if i == 0:
            pipeline.generator.model.set_quant_state(True, False)
            logger.info("Do Weight initialization!")
        elif i>=1 and i<=act_init_total_round:
            pipeline.generator.model.set_quant_state(True, True)
            logger.info("Collect Activations for calibration, and Do Activation initialization!")
        elif i> act_init_total_round  and i<= act_init_total_round + collected_recon_data_round: # collect fp
            pipeline.generator.model.set_quant_state(False, False)
            pipeline.generator.model.set_collect_recon_data(True)
            logger.info("Collect FP data for reconstruction training")
        elif i== act_init_total_round + collected_recon_data_round +1:
            pipeline.generator.model.set_collect_recon_data(False)
            reconstruct(pipeline.generator.model)
            logger.info("Save quantized checkpoint!")
            torch.save(pipeline.generator.model, os.path.join(exp_save_path, "quant_model.ckpt"))
            pipeline.generator.model.set_quant_state(True, True)
            break

        all_video = []
        num_generated_frames = 0  # Number of generated (latent) frames
        # For text-to-video, batch is just the text prompt
        prompt = data
        prompts = [prompt] * args.num_samples
        initial_latent = None
        curr_num_output_frames = args.num_output_frames
        sampled_noise = torch.randn(
            [args.num_samples, curr_num_output_frames, 16, 60, 104], device=device, dtype=torch.bfloat16
        )
        logger.info(prompts)
        # Generate 81 frames
        video, latents = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent,
            low_memory=False,
        )
        current_video = rearrange(video, 'b t c h w -> b t h w c').cpu()
        all_video.append(current_video)
        num_generated_frames += latents.shape[1]
        logger.info(f'video shape:{current_video.shape}, latent shape:{latents.shape}') ## [1, 81, 480, 832, 3], 
        # Final output video
        video = 255.0 * torch.cat(all_video, dim=1)
        # Clear VAE cache
        pipeline.vae.model.clear_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, help="Path to the config file")
    parser.add_argument("--weight_bit", type=int, default=8, help="int bit for weight quantization")
    parser.add_argument("--act_bit", type=int, default=8, help="int bit for activation quantization")
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument('--sensitivity', type=str)
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of overlap frames between sliding windows")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    args = parser.parse_args()
    main(args)