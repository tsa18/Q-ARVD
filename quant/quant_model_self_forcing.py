import logging
import torch
import torch.nn as nn
from quant.quant_block_self_forcing import get_specials, BaseQuantBlock, QuantCausalWanAttentionBlock, QuantCausalWanSelfAttention
from quant.quant_layer import QuantModule, StraightThrough
from wan.modules.causal_model import CausalWanAttentionBlock, CausalWanSelfAttention

logger = logging.getLogger(__name__)


class QuantModelSelfForcing(nn.Module):
    def __init__(self, model: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, **kwargs):
        super().__init__()
        self.model = model
        self.specials = get_specials()
        self.quant_module_refactor(self.model, weight_quant_params, act_quant_params)
        self.quant_block_refactor(self.model, weight_quant_params, act_quant_params)
        self.curr_quant_state = [False, False]
        

    def quant_module_refactor(self, module: nn.Module, weight_quant_params: dict = {}, act_quant_params: dict = {}, parent_name: str = ""):
        """
        Recursively replace the normal layers (conv2D, conv1D, Linear etc.) to QuantModule
        :param module: nn.Module with nn.Conv2d, nn.Conv1d, or nn.Linear in its children
        :param weight_quant_params: quantization parameters like n_bits for weight quantizer
        :param act_quant_params: quantization parameters like n_bits for activation quantizer
        """
        for name, child_module in module.named_children():
            full_module_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child_module, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                if 'time_embedding' in full_module_name or 'time_projection' in full_module_name \
                or 'head' in full_module_name: 
                    continue
                setattr(module, name, QuantModule(
                    child_module, weight_quant_params, act_quant_params, name=full_module_name))
            elif isinstance(child_module, StraightThrough):
                continue
            else:
                self.quant_module_refactor(child_module, weight_quant_params, act_quant_params, parent_name=full_module_name)

    def quant_block_refactor(self, module, weight_quant_params: dict = {}, act_quant_params: dict = {}):
        for name, child_module in module.named_children():
            if isinstance(child_module, CausalWanAttentionBlock):        
                setattr(module, name, QuantCausalWanAttentionBlock(child_module, act_quant_params))
            else:
                self.quant_block_refactor(child_module, weight_quant_params, act_quant_params)

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.curr_quant_state = [weight_quant, act_quant]
        for m in self.model.modules():
            if isinstance(m, (QuantModule, BaseQuantBlock)):
                m.set_quant_state(weight_quant, act_quant)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    

    def set_grad_ckpt(self, grad_ckpt: bool):
        for m in self.model.modules():
            if isinstance(m, (QuantCausalWanAttentionBlock)):
                m.checkpoint = grad_ckpt

    def set_curr_chunk_index(self, curr_chunk_index):
        for m in self.model.modules():
            if isinstance(m, (QuantModule)):
                m.set_curr_chunk_index(curr_chunk_index)

    def set_collect_recon_data(self, collect_recon_data: bool):
        for m in self.model.modules():
            # if isinstance(m, (QuantCausalWanAttentionBlock))
            if isinstance(m, (QuantModule)):
                m.set_collect_recon_data(collect_recon_data)


    def set_only_k_quant(self, k):
        for m in self.model.modules():
            if isinstance(m, (QuantModule)):
                m.set_only_k_quant(k)

    def set_sensitivity(self, sensitivity):
        for m in self.model.modules():
            if isinstance(m, (QuantModule)):
                m.set_sensitivity(sensitivity)

                