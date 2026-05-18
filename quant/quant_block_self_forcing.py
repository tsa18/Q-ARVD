import logging
import torch.nn as nn
from quant.quant_layer import QuantModule, UniformAffineQuantizer, StraightThrough
from wan.modules.causal_model import CausalWanAttentionBlock
import torch
import math
from wan.modules.attention import attention
from wan.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from typing import List, Tuple, Dict, Any
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from wan.modules.causal_model import causal_rope_apply
from wan.modules.attention import flash_attention

logger = logging.getLogger(__name__)


class BaseQuantBlock(nn.Module):
    """
    Base implementation of block structures for all networks.
    """
    def __init__(self, act_quant_params: dict = {}):
        super().__init__()
        self.use_weight_quant = False
        self.use_act_quant = False
        # initialize quantizer
        self.act_quantizer = UniformAffineQuantizer(**act_quant_params)
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # setting weight quantization here does not affect actual forward pass
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)



class QuantCausalWanAttentionBlock(BaseQuantBlock):
    def __init__(self, block, act_quant_params):
        super().__init__(act_quant_params)
        
        self.block = block
        # self.attn.use_act_quant = False
        # replace self-attn
        self.block.self_attn =  QuantCausalWanSelfAttention(self.block.self_attn, act_quant_params)
        self.block.cross_attn = QuantWanT2VCrossAttention(self.block.cross_attn, act_quant_params)
        self.block.ffn = QuantWanFFN(self.block.ffn, act_quant_params)

        ### record data for reconstruction
        ### 重构数据记录相关变量（完善初始化）
        self.collect_recon_data = False  # 记录开关
        self.collected_input_args: List[Tuple[Any, ...]] = []  # 记录位置参数
        self.collected_input_kwargs: List[Dict[str, Any]] = []  # 记录关键字参数
        # 可选：记录输出（如果重构需要）
        self.collected_outputs: List[torch.Tensor] = []



    # reuse the original forward
    def forward(self,*args, **kwargs):
         # 当开启记录时，保存输入（detach避免梯度关联，clone防止数据被覆盖）
        if self.collect_recon_data:
            # 对tensor类型的参数进行detach+clone，非tensor直接保存
            saved_args = []
            for arg in args:
                if isinstance(arg, torch.Tensor):
                    saved_args.append(arg.detach().cpu())
                else:
                    saved_args.append(arg)
            self.collected_input_args.append(tuple(saved_args))
            
            saved_kwargs = {}
            for k, v in kwargs.items():
                if isinstance(v, torch.Tensor) and 'freqs' not in k:
                    saved_kwargs[k] = v.detach().cpu()
                    print(f"Collecting input for reconstruction - {k}: shape {v.shape}")
                else:
                    saved_kwargs[k] = v
                    print(f"Collecting input for reconstruction - {k}: non-tensor or excluded from detaching")
            self.collected_input_kwargs.append(saved_kwargs)

         
        # --------------------- forward -----------------
        output = self.block(*args, **kwargs)
        # -----------------------------------------------

         # 可选：记录输出（如果重构需要对比输出）
        if self.collect_recon_data:
            assert isinstance(output, torch.Tensor), "Output is not a tensor"
            self.collected_outputs.append(output.detach().cpu())



        return output

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        # self.attn.use_act_quant = act_quant 控制BMM
        for m in self.modules():
            if isinstance(m, QuantModule):
                m.set_quant_state(weight_quant, act_quant)

    def set_collect_recon_data(self, collect_recon_data: bool):
        self.collect_recon_data = collect_recon_data


class QuantCausalWanSelfAttention(nn.Module):
    def __init__(self, attn, act_quant_params):
        super().__init__()
        
        self.attn = attn
        self.use_act_quant = False

    def __getattr__(self, name):
        """
        属性委托方法:
        __getattr__ 会在访问一个不存在的属性时被调用
        当自身找不到属性时，去 self.attn 中查找
        """
        try:
        # 第一步：尝试按照 PyTorch 标准规则在当前类（Quant类）里找
        # 这确保了 self.act_quant_params 能被正常访问
            return super().__getattr__(name)
        except AttributeError:
        # 第二步：如果当前类确实没有，说明可能是原始 attn 里的属性
        # 比如 self.attn.head_dim 或者 self.attn.q_proj
            return getattr(self.attn, name)
    
    ## override orignial forward to enable smooth
    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        # print(111111, x.shape) ### [1, 4680, 1536])
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            frame_seqlen = math.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(
                q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(
                k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)

            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_query.shape[1]
            if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                    kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                # Insert the new keys/values at the end
                local_end_index = kv_cache["local_end_index"].item() + current_end - \
                    kv_cache["global_end_index"].item() - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x




class QuantWanT2VCrossAttention(nn.Module):
    def __init__(self, attn, act_quant_params):
        super().__init__()
        
        self.cross_attn = attn
        self.use_act_quant = False

    def __getattr__(self, name):
        """
        属性委托方法:
        __getattr__ 会在访问一个不存在的属性时被调用
        当自身找不到属性时，去 self.attn 中查找
        """
        try:
        # 第一步：尝试按照 PyTorch 标准规则在当前类（Quant类）里找
        # 这确保了 self.act_quant_params 能被正常访问
            return super().__getattr__(name)
        except AttributeError:
        # 第二步：如果当前类确实没有，说明可能是原始 attn 里的属性
        # 比如 self.attn.head_dim 或者 self.attn.q_proj
            return getattr(self.cross_attn, name)
    
    ## override orignial forward to enable smooth
    def forward(self, x, context, context_lens, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x



class QuantWanFFN(nn.Module):
    def __init__(self, ffn, act_quant_params):
        super().__init__()
        
        self.ffn = ffn
        self.use_act_quant = False


    def __getattr__(self, name):
        """
        属性委托方法:
        __getattr__ 会在访问一个不存在的属性时被调用
        当自身找不到属性时，去 self.attn 中查找
        """
        try:
        # 第一步：尝试按照 PyTorch 标准规则在当前类（Quant类）里找
        # 这确保了 self.act_quant_params 能被正常访问
            return super().__getattr__(name)
        except AttributeError:
        # 第二步：如果当前类确实没有，说明可能是原始 attn 里的属性
        # 比如 self.attn.head_dim 或者 self.attn.q_proj
            return getattr(self.ffn, name)

    def forward(self, x):

        x = self.ffn[0](x) # linear
        x = self.ffn[1](x) # GeLU
        x = self.ffn[2](x) # linear
        return x


    




def get_specials():
    specials = {
        CausalWanAttentionBlock: QuantCausalWanAttentionBlock,
    }
    return specials
