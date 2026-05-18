import logging
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
import numpy as np
import gc
import math

logger = logging.getLogger(__name__)


def print_out_channel_max_min(name, max_tensor: torch.Tensor, min_tensor: torch.Tensor):

    print(f"\n===== {name} 逐通道原始最大/最小值 =====")
    for idx, (max_val, min_val) in enumerate(zip(max_tensor, min_tensor)):
        print(f"out-channel [{idx}]: max = {max_val:.6f} | min = {min_val:.6f}")





class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()


class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.
    :param n_bits: number of bit for quantization
    :param channel_wise: if True, compute scale and zero_point in each channel
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False, scale_method: str = 'max',
                 leaf_param: bool = False, always_zero: bool = False, name='undefined', qtype='none', use_dual_scale=False):
        super(UniformAffineQuantizer, self).__init__()
        assert 2 <= n_bits <= 32, 'bitwidth not supported'
        self.sym = symmetric
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits if not self.sym else 2 ** (self.n_bits - 1) - 1
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.channel_wise = channel_wise
        self.leaf_param = leaf_param
        self.scale_method = scale_method
        self.running_stat = False
        self.always_zero = always_zero
        
        if 'cross_attn.k' in name or 'cross_attn.v' in name:  
            self.min_init_batch_size = 1 # one prompt
        else:
            self.min_init_batch_size = 1*(4+1)*7 ## prompt*(timestep+1)*chunks
            # self.min_init_batch_size = 1*(4+1)*11 ## prompt*(timestep+1)*chunks
        self.current_bs = 0
        self.collected_batch = []
        self.collected_chunk_id = []
        self.collected_time = []
        self.use_dual_scale = use_dual_scale
        self.is_out_layer = False


        if self.leaf_param:
            self.x_min, self.x_max = None, None
        self.name = name
        self.qtype = qtype

        self.curr_chunk_index = 0
        self.curr_time = 0

        self.sensitivity = None
    
    def __repr__(self):
        s = super(UniformAffineQuantizer, self).__repr__()
        s = "(" + s + " inited={}, channel_wise={})".format(self.inited, self.channel_wise)
        return s

    def forward(self, x: torch.Tensor):
        assert self.qtype!='none', "Quantizer type not specified!"
        ### if it is the act quantizer
        if not self.inited:
            if self.qtype == 'act':
                self.current_bs = self.current_bs + 1
                self.collected_batch.append(x.detach().cpu())
                self.collected_chunk_id.append(self.curr_chunk_index)
                self.collected_time.append(self.curr_time)
                # logger.info(f"collected shape:{x.shape}, name:{self.name}")
                if self.current_bs == self.min_init_batch_size:
                    # logger.info(f'start init act quant: {self.name}, round {self.curr_init_round}')
                    logger.info(f'start init act quant: {self.name}')
                    x_for_init = torch.cat(self.collected_batch, dim=0).to(x)

                    # ========== Chunk 加权 ==========
                    chunk_ids = torch.tensor(self.collected_chunk_id, dtype=torch.int)
                    raw_chunk_weights = self.sensitivity if self.sensitivity is not None else [1,1,1,1,1,1,1]
                    mean_raw = sum(raw_chunk_weights) / len(raw_chunk_weights)
                    scaled_chunk_weights = [w / mean_raw for w in raw_chunk_weights]
                    weight_tensor_chunk = torch.tensor(scaled_chunk_weights)
                    sample_weights_chunk = weight_tensor_chunk[chunk_ids].to(x)  # shape: [N]
       
                    sample_weights = sample_weights_chunk

                    if self.use_dual_scale or self.is_out_layer:
                    # if True: ## when get weights/ori
                        sample_weights = torch.ones_like(sample_weights)

                    if self.leaf_param:
                        delta, self.zero_point = self.init_quantization_scale(x_for_init, self.channel_wise, sample_weights=sample_weights)
                        self.delta = torch.nn.Parameter(delta)
                    else:
                        self.delta, self.zero_point = self.init_quantization_scale(x_for_init, self.channel_wise, sample_weights=sample_weights)
                    del x_for_init, self.collected_batch
                    import gc
                    gc.collect()
                    torch.cuda.empty_cache()
                    self.current_bs = 0
                    self.collected_batch = []
                    self.collected_chunk_id = []
                    self.collected_time = []
                    self.inited = True
                return x

            else:
                if self.leaf_param:
                    delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                    self.delta = torch.nn.Parameter(delta)
                else:
                    self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                    self.delta = torch.nn.Parameter(self.delta)
                self.inited = True
    

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        if self.sym:
            x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
        else:
            x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        # x_quant =  x_int # no clip error          
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant


    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False, sample_weights: torch.Tensor=None):
        delta, zero_point = None, None
        x = x.float() # avoid unsupported bf16 for quantile, ensure scale is fp32
        ### fast parallel weight quantization init：
        if channel_wise:
            logger.info('init weight')
            x_clone = x.clone().detach()
            # 1. 统一张量形状：(n_channels, -1)
            if len(x.shape) == 4:  # Conv2d权重 [out_ch, in_ch, h, w]
                x_reshaped = x_clone.reshape(x_clone.shape[0], -1)  # [n_ch, flatten]
            elif len(x.shape) == 2:  # Linear权重 [out_ch, in_ch]
                x_reshaped = x_clone  # 已为[ch, feat]
            elif len(x.shape) == 3:  # 如Conv1d [out_ch, in_ch, len]
                x_reshaped = x_clone.reshape(x_clone.shape[0], -1)
            else:
                raise NotImplementedError

            # 2. 批量计算所有通道的min/max（或quantile）
            if 'max' in self.scale_method:
                x_min = x_reshaped.min(dim=1)[0]  # [n_ch]
                x_max = x_reshaped.max(dim=1)[0]  # [n_ch]
                if self.sym:
                    x_absmax = torch.max(x_min.abs(), x_max.abs())  # [n_ch]
                    delta = x_absmax / self.n_levels  # 批量计算delta
                else:
                    delta = (x_max - x_min) / (self.n_levels - 1)  # 批量计算delta
                delta = torch.clamp(delta, min=1e-8)  # 批量clamp
                # 批量计算zero_point
                zero_point = (-x_min / delta).round() if not (self.sym or self.always_zero) else torch.zeros_like(delta)
            else:
                # 若用quantile方法，同样批量计算所有通道的quantile
                x_flat = x_reshaped.reshape(x_reshaped.shape[0], -1)  # [n_ch, all_elem]
                best_scores = torch.ones(x_reshaped.shape[0]).to(x) * 1e10  # [n_ch]
                delta = torch.zeros(x_reshaped.shape[0]).type_as(x)
                zero_point = torch.zeros(x_reshaped.shape[0]).type_as(x)
                # 批量计算各通道的最优quantile
                for pct in [0.999, 0.9999, 0.99999]:
                    try:
                        new_max = torch.quantile(x_flat, pct, dim=1)  # [n_ch]
                        new_min = torch.quantile(x_flat, 1-pct, dim=1)  # [n_ch]                   
                    except:
                        print(123)
                        # 兼容numpy percentile（批量）
                        x_np = x_flat.cpu().numpy()
                        new_max = torch.tensor(np.percentile(x_np, pct*100, axis=1), device=x.device)
                        new_min = torch.tensor(np.percentile(x_np, (1-pct)*100, axis=1), device=x.device)
                    # print_out_channel_max_min(self.name, new_max, new_min)
                    # 批量量化+计算loss
                    if self.sym:
                        delta_pct = torch.max(new_max.abs(), new_min.abs()) / self.n_levels
                        zp_pct = torch.zeros_like(delta_pct)
                        x_int = torch.round(x_flat / delta_pct.unsqueeze(1))
                        x_quant = torch.clamp(x_int, -self.n_levels - 1, self.n_levels)
                        x_dequant = (x_quant) * delta_pct.unsqueeze(1)
                    else:
                        delta_pct = (new_max - new_min) / (2**self.n_bits - 1) # [n_ch]
                        zp_pct = (-new_min / delta_pct).round() # [n_ch]
                        x_int = torch.round(x_flat / delta_pct.unsqueeze(1))
                        x_quant = torch.clamp(x_int + zp_pct.unsqueeze(1), 0, self.n_levels-1)
                        x_dequant = (x_quant - zp_pct.unsqueeze(1)) * delta_pct.unsqueeze(1)

                    # 批量计算lp_loss
                    # score = lp_loss(x_flat, x_dequant, p=2, reduction='none').mean(dim=1)  # [n_ch]
                    score =  (x_flat-x_dequant).abs().pow(2).mean(dim=1).float()  # [n_ch]
                    # 批量更新最优值
                    update_mask = score < best_scores
                    best_scores[update_mask] = score[update_mask]
                    delta[update_mask] = delta_pct[update_mask]
                    zero_point[update_mask] = zp_pct[update_mask]
                

            # 3. 恢复delta/zero_point的维度形状（与原代码一致）
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            elif len(x.shape) == 2:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
            elif len(x.shape) == 3:
                delta = delta.view(-1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1)

                
        else: # non-channel-wise per-tensor, asysmetric
            # logger.info('init activation')
            assert not self.sym
            if self.leaf_param:
                self.x_min = x.data.min()
                self.x_max = x.data.max()

            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    delta = x_absmax / self.n_levels
                else:
                    delta = float(x.max().item() - x.min().item()) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta) if not (self.sym or self.always_zero) else 0
                delta = torch.tensor(delta).type_as(x)
            else:
              

                x_clone = x
                x_max = x_clone.max()
                x_min = x_clone.min()
                batch_chunk_size = 5 ## TODO 需要能整除样本数, 否则最后一个batch 可能size 比较小，影响 mean
                assert (x.shape[0]% batch_chunk_size ==0 or x.shape[0]< batch_chunk_size )
                total_batch = x_clone.shape[0]  # 总batch数
                num_chunks = (total_batch + batch_chunk_size - 1) // batch_chunk_size
                best_score = 1e+10
                
                num_bins = 32768
                hist = torch.histc(x_clone, bins=num_bins, min=x_min, max=x_max)
                cdf = torch.cumsum(hist, dim=0)
                cdf = cdf / cdf[-1]  # 归一化到 [0, 1]
                bin_width = (x_max - x_min) / num_bins

                for pct in [0.999, 0.9999, 0.99999]:
                    # 基于全局CDF找分位数对应的索引
                    idx_min = torch.searchsorted(cdf, 1.0 - pct)
                    idx_max = torch.searchsorted(cdf, pct)
                    # 计算new_min/new_max
                    new_min = x_min + (idx_min.float() + 0.5) * bin_width
                    new_max = x_min + (idx_max.float() + 0.5) * bin_width 
                    # 分batch chunk
                    total_score = 0.0
                    for i in range(num_chunks):
                        start_idx = i * batch_chunk_size
                        end_idx = min((i + 1) * batch_chunk_size, total_batch)
                        x_batch_chunk = x_clone[start_idx:end_idx]
                        x_q = self.quantize(x_batch_chunk, new_max, new_min)
                        # score = lp_loss(x_batch_chunk, x_q, p=2, reduction='all')
                        score = (x_batch_chunk-x_q).abs().pow(2).view(x_batch_chunk.shape[0], -1).mean(dim=1)
                        score = ( sample_weights[start_idx:end_idx] * score ).mean()
                        total_score += score.item()                        
                    
                    avg_score = total_score / num_chunks
                    if avg_score < best_score:
                        best_score = avg_score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()


        return delta, zero_point
    
    def quantize(self, x, max, min):
        if self.sym:
            delta = torch.max(max.abs(), min.abs()) / self.n_levels
            zero_point = torch.zeros_like(delta)
            lower, upper = -self.n_levels - 1, self.n_levels
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int, lower, upper)
            x_float_q = (x_quant) * delta   
        else:
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    
    def eff_mem_quantize(self, x, max, min):
        if self.sym:
            delta = torch.max(max.abs(), min.abs()) / self.n_levels
            zero_point = torch.zeros_like(delta)
            lower, upper = -self.n_levels - 1, self.n_levels
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int, lower, upper)
            x_float_q = (x_quant) * delta   
        else:
            delta = (max - min) / (2 ** self.n_bits - 1)
            zero_point = (- min / delta).round()
            x_int = torch.round(x / delta)
            x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
            x_float_q = (x_quant - zero_point) * delta
        return x_float_q
    


    

class QuantModule(nn.Module):
    """
    Quantized Module that can perform quantized convolution or normal convolution.
    To activate quantization, please use set_quant_state function.
    """
    def __init__(self, org_module: Union[nn.Linear], weight_quant_params: dict = {},
                 act_quant_params: dict = {}, disable_act_quant: bool = False, name=""):
        super(QuantModule, self).__init__()
        self.weight_quant_params = weight_quant_params
        self.act_quant_params = act_quant_params
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        elif isinstance(org_module, nn.Conv1d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv1d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
        # self.weight = org_module.weight.data
        self.weight = org_module.weight 
        if org_module.bias is not None:
            self.bias = org_module.bias.data
        else:
            self.bias = None
        # de-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False
        self.disable_act_quant = disable_act_quant
        # initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(**self.weight_quant_params, name=name, qtype='weight')
        self.act_quantizer = UniformAffineQuantizer(**self.act_quant_params, name=name, qtype='act')

        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

        self.extra_repr = org_module.extra_repr
        self.name = name

        self.curr_chunk_index = 0
        self.curr_time = 0

        self.collect_recon_data = False
        self.collected_Xs = []
        self.collected_Ys = []

        # ---- SmoothQuant ----
        self.use_smooth_quant = False
        # smooth_scale shape: [in_channels]，存为 buffer（不参与梯度/优化，但会随模型保存）
        self.register_buffer('smooth_scale', None)
        self.collect_smooth_data = False
        # self.act_max = None
        self.act_max = {}

        self.only_quant_chunk_id = None
        self.only_quant_time_t = None
        
        self.smooth_delta = None
        self.use_dual_scale = False
        self.is_out_layer = False

        self.sensitivity = None

    def forward(self, input: torch.Tensor):

        ori_dtype = input.dtype

        def cuda_time(label, func):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            result = func()
            e.record()
            torch.cuda.synchronize()
            print(f"[{label}] {s.elapsed_time(e):.3f} ms")
            return result
        
        if not self.use_dual_scale:
            weight, input = self.weight, input
            if self.only_quant_chunk_id is None or (self.curr_chunk_index == self.only_quant_chunk_id):
                if not self.disable_act_quant and self.use_act_quant:
                    # input = cuda_time("act_quantizer", lambda: self.act_quantizer(input.float()))
                    input = self.act_quantizer(input.float())
                if self.use_weight_quant:
                    weight = self.weight_quantizer(self.weight.float())
            weight =  weight.to(ori_dtype)
            input  = input.to(ori_dtype)
            out = self.fwd_func(input, weight, self.bias, **self.fwd_kwargs)
            # out = cuda_time("fwd_func XW", lambda: self.fwd_func(input, weight, self.bias, **self.fwd_kwargs))
        else:
            W1, W2 = self.W_outlier, self.W_normal
            X1 = input[..., self.outlier_indices]
            X2 = input[..., self.normal_indices]
            if self.only_quant_chunk_id is None or (self.curr_chunk_index == self.only_quant_chunk_id):
                if not self.disable_act_quant and self.use_act_quant:
                    X1 = self.act_quantizer_outlier(X1.float())
                    X2 = self.act_quantizer_normal(X2.float())
                if self.use_weight_quant:
                    W1 = self.weight_quantizer_outlier(W1.float())
                    W2 = self.weight_quantizer_normal(W2.float())
            W1, W2 = W1.to(ori_dtype), W2.to(ori_dtype)
            X1, X2 = X1.to(ori_dtype), X2.to(ori_dtype)
            # 加这几行
            out1 = self.fwd_func(X1, W1, self.bias, **self.fwd_kwargs)
            out2 = self.fwd_func(X2, W2, None, **self.fwd_kwargs)  # 只加一次bias，避免重复
            # out2 = cuda_time("Dual fwd_func X2W2", lambda: self.fwd_func(X2, W2, self.bias, **self.fwd_kwargs))
            out = out1 + out2


        out = self.activation_function(out)
        if self.collect_recon_data:
            assert not self.use_act_quant and not self.use_weight_quant
            self.collected_Xs.append((input.detach().cpu(), self.curr_chunk_index, self.curr_time)) # fp input
            self.collected_Ys.append((out.detach().cpu(), self.curr_chunk_index, self.curr_time))  # fp output

        return out



    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_running_stat(self, running_stat: bool):
        if not self.use_dual_scale:
            self.act_quantizer.running_stat = running_stat
        else:
            self.act_quantizer_outlier.running_stat = running_stat
            self.act_quantizer_normal.running_stat = running_stat

    def set_curr_chunk_index(self, curr_chunk_index):
        self.curr_chunk_index = curr_chunk_index
        if not self.use_dual_scale:
            self.weight_quantizer.curr_chunk_index = curr_chunk_index
            self.act_quantizer.curr_chunk_index = curr_chunk_index
        else:
            self.weight_quantizer_outlier.curr_chunk_index = curr_chunk_index
            self.weight_quantizer_normal.curr_chunk_index = curr_chunk_index
            self.act_quantizer_outlier.curr_chunk_index = curr_chunk_index
            self.act_quantizer_normal.curr_chunk_index = curr_chunk_index


    def set_curr_time(self, t):
        self.curr_time = t
        if not self.use_dual_scale:
            self.weight_quantizer.curr_time = t
            self.act_quantizer.curr_time = t
        else:
            self.weight_quantizer_outlier.curr_time = t
            self.weight_quantizer_normal.curr_time = t
            self.act_quantizer_outlier.curr_time = t
            self.act_quantizer_normal.curr_time = t



    def set_collect_recon_data(self, collect_recon_data: bool):
        self.collect_recon_data = collect_recon_data

    def set_collect_smooth_data(self, collect_smooth_data: bool):
        self.collect_smooth_data = collect_smooth_data

    def set_only_k_quant(self, k):
        self.only_quant_chunk_id = k

    def set_only_time_t_quant(self, t):
        self.only_quant_time_t = t

    def set_sensitivity(self, sensitivity):
        self.sensitivity = sensitivity
        if not self.use_dual_scale:
            self.weight_quantizer.sensitivity = sensitivity
            self.act_quantizer.sensitivity = sensitivity
        else:
            self.weight_quantizer_outlier.sensitivity = sensitivity
            self.weight_quantizer_normal.sensitivity = sensitivity
            self.act_quantizer_outlier.sensitivity = sensitivity
            self.act_quantizer_normal.sensitivity = sensitivity

        
