import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from typing import Literal, Optional
from quant.quant_layer import QuantModule
import logging
from quant.quant_layer import UniformAffineQuantizer

logger = logging.getLogger(__name__)


def _plot_layer_outliers(
    ax: plt.Axes,
    name: str,
    norms_np: np.ndarray,
    outlier_indices: np.ndarray,
    threshold: Optional[float] = None,
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

    # Threshold line: use the externally provided threshold directly
    if threshold is not None:
        ax.axhline(threshold, color=C_THRESH, linewidth=0.9,
                   linestyle=":", label=f"threshold={threshold:.3f}", zorder=3)

    ax.legend(fontsize=6, loc="upper right", framealpha=0.7)
    ax.set_xlabel("In-Channel (sorted by norm)", fontsize=8)
    ax.set_ylabel("L2 Norm", fontsize=8)
    ax.set_title(short, fontsize=9, pad=4)
    ax.tick_params(labelsize=7)
    ax.set_xlim(-n_ch * 0.02, n_ch)
    ax.set_ylim(bottom=0)


def detect_outlier_count(
    col_norms: torch.Tensor,
    method: Literal["iqr", "mad_zscore"] = "iqr",
    align: int = 32,
    iqr_k: float = 1.5,
    mad_threshold: float = 3.5,
    min_ratio_vs_median: float = 1.2,
) -> tuple[int, float]:
    """
    Automatically determine the number of outlier channels based on column norms.

    Args:
        col_norms: L2 norm per input channel, shape [in_features].
        method: Detection method ('iqr' or 'mad_zscore').
        align: Round up n_outlier to a multiple of this value (hardware-friendly; set to 1 to disable).
        iqr_k: IQR multiplier coefficient (1.5 = standard outlier, 3.0 = extreme outlier).
        mad_threshold: MAD Z-score threshold (recommended 3.5).
        min_ratio_vs_median: Minimum ratio vs. median to avoid false positives on flat distributions.
    Returns:
        n_outlier: Number of outlier channels (aligned, minimum 0).
        threshold: The effective threshold value used for detection.
    """
    norms_np = col_norms.float().cpu().numpy()
    n = len(norms_np)

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
        if mad < 1e-8:          # all values nearly identical, no outliers
            return 0, 0.0
        modified_z = 0.6745 * (norms_np - median) / mad
        n_outlier = int(np.sum(
            (modified_z > mad_threshold) & (norms_np > min_ratio_vs_median * median)
        ))

        mad_based_threshold = median + (mad_threshold * mad) / 0.6745
        threshold = max(mad_based_threshold, min_ratio_vs_median * median)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Round up to nearest multiple of align (0 stays 0)
    if n_outlier > 0 and align > 1:
        n_outlier = (n_outlier + align - 1) // align * align

    # Safety: cap at 50% of total channels
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

    # When visualize=True, collect plot data first, then render all at once
    vis_records: list[dict] = []
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

        # Collect plot data before weight splitting
        if visualize:
            vis_records.append({
                "name": name,
                "norms_np": col_norms.cpu().numpy(),
                "outlier_indices": outlier_indices.cpu().numpy(),
                "threshold": threshold,
            })

        # Split weight into outlier and normal sub-matrices
        module.outlier_indices = outlier_indices
        module.normal_indices  = normal_indices
        module.use_dual_scale  = True
        module.W_outlier = module.weight[:, outlier_indices].to(dtype=dtype, device=device).contiguous()
        module.W_normal  = module.weight[:, normal_indices ].to(dtype=dtype, device=device).contiguous()
        module.weight    = None

        # Split quantizers: create separate outlier/normal quantizers for both weight and activation
        for qtype, attr_orig in [("weight", "weight_quantizer"), ("act", "act_quantizer")]:
            orig_q      = getattr(module, attr_orig)
            orig_params = getattr(module, f"{qtype}_quant_params")
            setattr(module, f"{attr_orig}_outlier", UniformAffineQuantizer(
                **orig_params, name=f"{orig_q.name}_outlier", qtype=qtype, use_dual_scale=True))
            setattr(module, f"{attr_orig}_normal", UniformAffineQuantizer(
                **orig_params, name=f"{orig_q.name}_normal", qtype=qtype, use_dual_scale=True))
            setattr(module, attr_orig, None)

        logger.info(
            f"{name}: method={method}, n_outlier={n_outlier}/{col_norms.shape[0]} "
            f"({100 * n_outlier / col_norms.shape[0]:.1f}%)"
        )

    # Render all outlier detection plots at once
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