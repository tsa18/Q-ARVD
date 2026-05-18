<p align="center">
<h1 align="center">Q-ARVD</h1>
<h3 align="center">Quantizing Autoregressive Video Diffusion Models</h3>
</p>
<p align="center">
  <p align="center">
    <a href="https://scholar.google.com/citations?user=0LBP5eIAAAAJ&hl=en">Siao Tang</a><sup>1</sup>
    ·
    <a href="https://horseee.github.io/">Xinyin Ma</a><sup>1</sup>
    ·
    <a href="https://fangggf.github.io/">Gongfan Fang</a><sup>1</sup>
    ·
    <a href="https://adamdad.github.io/">Xingyi Yang</a><sup>2</sup>
    ·
    <a href="https://scholar.google.com/citations?user=w69Buq0AAAAJ&hl=en">Xinchao Wang</a><sup>1</sup><br>
    <sup>1</sup>National University of Singapore &nbsp; <sup>2</sup>The Hong Kong Polytechnic University</p>
  <h3 align="center"><a href="https://arxiv.org/abs/2506.08009">Paper</a></h3>
</p>

---

Q-ARVD proposes the first quantization framework tailored for autoregressive video diffusion models. It introduces a final-quality guided frame-weighting mechanism to handle the unbalanced frame-wise quantization sensitivity, and an outlier-aware adaptive dual-scale strategy to address the heterogeneous outlier patterns. 

<p align="center">
  <img src="assets/framework.jpg" alt="Q-ARVD Framework" width="90%">
</p>

---


## Installation
Create a conda environment and install dependencies:
```
conda create -n q_arvd python=3.10 -y
conda activate q_arvd
# Set up self-forcing environment
## pytorch
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
## flash attention
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
pip install flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
## others
pip install -r requirements.txt
## Vbench
pip install vbench
pip install detectron2@git+https://github.com/facebookresearch/detectron2.git # require pytorch version with CUDA<=12.1

```

## Quick Start
### Download self-forcing checkpoints
```
hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
hf download gdhe17/Self-Forcing checkpoints/self_forcing_dmd.pt --local-dir .
```

### Step 1. Get Chunk-wise Quantization Sensitivity

```
bash scripts/get_chunkwise_sensitivity.sh
```
After completing the script, you will obtain a chunk-wise sensitivity like [0.5462, 0.1668, 0.1189, 0.0789, 0.0534, 0.0263, 0.0096], which will be used in the quantization process (Step2).

### Step 2. Start Quantization and Save Quantized Ckpt

```
bash scripts/train_quantization.sh
```


### Step 3. Generate Samples with Quantized Model

```
# 1. generate reference images with bfloat16 model
bash scripts/infer_fp_model.sh
# 2. generate quantized images with quantized model
bash scripts/infer_quant_model.sh
```

### Step 4. Calculate Vbench, FVD-FP, and LPIPS-FP Metrics
```
bash scripts/eval_quant_model.sh
```


## Acknowledgements
This codebase is built on top of the open-source implementations, including [Self-forcing](https://github.com/guandeh17/Self-Forcing), [PTQ4ViT](https://github.com/adreamwu/PTQ4DiT), [BRECQ](https://github.com/yhhhli/BRECQ), and [common_metrics_on_video_quality](https://github.com/JunyaoHu/common_metrics_on_video_quality), etc.



## Citation
If you find this codebase useful for your research, please kindly cite our paper:
```
@article{huang2025selfforcing,
  title={Self Forcing: Bridging the Train-Test Gap in Autoregressive Video Diffusion},
  author={Huang, Xun and Li, Zhengqi and He, Guande and Zhou, Mingyuan and Shechtman, Eli},
  journal={arXiv preprint arXiv:2506.08009},
  year={2025}
}
```
