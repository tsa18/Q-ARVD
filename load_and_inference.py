# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample new images from a pre-trained DiT.
"""
import torch
import torch.nn as nn
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
from quant.quant_layer import UniformAffineQuantizer, QuantModule
from quant.adaptive_rounding import AdaRoundQuantizer


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
    
    print(f"Inference Arguments: {args}")
    # Setup PyTorch:
    torch.set_grad_enabled(False)

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


    #### 2. Load quantized model ckpt
    logger.info('================== Loading quant model ==================')
    if args.use_quant:
        pipeline.generator.model = torch.load(args.quant_ckpt_path, map_location=gpu)
        pipeline.generator.model.to(device=gpu)
        # enable quantization
        pipeline.generator.model.set_quant_state(True, True)
        # skip init
        for m in pipeline.generator.model.modules():
            if isinstance(m, UniformAffineQuantizer):
                m.inited = True
            elif isinstance(m, AdaRoundQuantizer):
                m.soft_targets = False
            elif isinstance(m, QuantModule): # compatible
                if not hasattr(m, 'use_dual_scale'):
                    setattr(m, 'use_dual_scale', False)


    #### 3. Inference with quantized model
    # Create dataset
    with open(args.data_path, 'r') as f:
        lines = f.readlines()
        all_prompts = [line.strip() for line in lines]
    output_path = os.path.join("test_out", args.test_name)
    os.makedirs(output_path, exist_ok=True)

    # for _ in range(10000000000):
    for _ in range(1):
        for i, data in tqdm(enumerate(all_prompts), disable=(local_rank != 0)):
            logger.info("Generate videos with quantized model with lora reconstruction!")

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

            # Save the video if the current prompt is not a dummy prompt
            for seed_idx in range(args.num_samples):
                video_path = os.path.join(output_path, f'{prompt[:100]}-{seed_idx}-{i}.mp4')
                write_video(video_path, video[seed_idx], fps=16)
            
            if i >= args.gen_num - 1:
                break




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_name", type=str, help="Path to the config file")
    parser.add_argument("--quant_ckpt_path", type=str, help="Path to the quantized checkpoint")
    # inference
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of overlap frames between sliding windows")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--gen_num", type=int, default=50, help="Number of videos to generate")
    parser.add_argument("--use_quant", action="store_true")
    


    args = parser.parse_args()
    main(args)
