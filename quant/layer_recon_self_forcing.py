import torch
import logging
from quant.quant_layer import QuantModule, StraightThrough, lp_loss
from quant.adaptive_rounding import AdaRoundQuantizer
import torch.nn.functional as F


logger = logging.getLogger(__name__)


def reconstruct(model):
    """
    Reconstruct model
    """
    logger.info('-------------  start reconstruction ------------- ')
    torch.set_grad_enabled(True)
    for name, module in model.named_modules():
        if isinstance(module, QuantModule):
            if 'cross_attn.k' in name or 'cross_attn.v' in name or 'text_embedding' in name:
                continue
            logger.info(f"Reconstruction for layer: {name}")
            layer_reconstruction(model, module, weight=0.01, warmup=0.2, iters=2000, batch_size=8, lr_w=2e-3, lr=4e-5)
    logger.info('------------- finish reconstruction ------------- ')
    torch.set_grad_enabled(False)


def layer_reconstruction(model, layer: QuantModule,
                         batch_size: int = 32, iters: int = 20000, weight: float = 0.001, opt_mode: str = 'mse',
                         asym: bool = False, include_act_func: bool = True, b_range: tuple = (20, 2),
                         warmup: float = 0.0, act_quant: bool = False, lr_w=2e-3, lr: float = 4e-5, p: float = 2.0,
                         outpath: str = None):
    """
    Block reconstruction to optimize the output from each layer.
    :param model: QuantModel
    :param layer: QuantModule that needs to be optimized
    :param batch_size: mini-batch size for reconstruction
    :param iters: optimization iterations for reconstruction,
    :param weight: the weight of rounding regularization term
    :param opt_mode: optimization mode
    :param asym: asymmetric optimization designed in AdaRound, use quant input to reconstruct fp output
    :param include_act_func: optimize the output after activation function
    :param b_range: temperature range
    :param warmup: proportion of iterations that no scheduling for temperature
    :param act_quant: use activation quantization or not.
    :param lr: learning rate for act delta learning
    :param p: L_p norm minimization
    :param multi_gpu: use multi-GPU or not, if enabled, we should sync the gradients
    :param cond: conditional generation or not
    """
    # Set up quantization state:
    model.set_quant_state(False, False)
    layer.set_quant_state(True, True)

    if not include_act_func:
        org_act_func = layer.activation_function
        layer.activation_function = StraightThrough()

    # Replace weight quantizer to AdaRoundQuantizer:
    round_mode = 'learned_hard_sigmoid'
    if not layer.use_dual_scale:
        layer.weight_quantizer = AdaRoundQuantizer(uaq=layer.weight_quantizer, round_mode=round_mode, weight_tensor=layer.weight)
        layer.weight_quantizer.soft_targets = True
    else:
        layer.weight_quantizer_outlier = AdaRoundQuantizer(uaq=layer.weight_quantizer_outlier, round_mode=round_mode, weight_tensor=layer.W_outlier)
        layer.weight_quantizer_normal = AdaRoundQuantizer(uaq=layer.weight_quantizer_normal, round_mode=round_mode, weight_tensor=layer.W_normal)
        layer.weight_quantizer_outlier.soft_targets, layer.weight_quantizer_normal.soft_targets = True, True
        

    # Set up optimizer:
    if not layer.use_dual_scale:
        opt_params_w = [layer.weight_quantizer.alpha]
    else:
        opt_params_w = [layer.weight_quantizer_outlier.alpha, layer.weight_quantizer_normal.alpha]

    optimizer_w = torch.optim.Adam(opt_params_w, lr=lr_w)
    if not layer.use_dual_scale:
        opt_params_a = [layer.act_quantizer.delta]
    else:
        opt_params_a = [layer.act_quantizer_outlier.delta, layer.act_quantizer_normal.delta]
    optimizer_a = torch.optim.Adam(opt_params_a, lr=lr)
    scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_a, T_max=iters, eta_min=0.)

    # Set up loss function:
    loss_mode = 'relaxation'
    rec_loss = opt_mode
    loss_func = LossFunction(layer, round_loss=loss_mode, weight=weight,
                             max_count=iters, rec_loss=rec_loss, b_range=b_range,
                             decay_start=0, warmup=warmup, p=p)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_list = []
    chunk_ids_list = []
    time_list = []
    for x, chunk_id, time in layer.collected_Xs:
        X_list.append(x)
        chunk_ids_list.append(torch.full((x.shape[0],), chunk_id, dtype=torch.long))
        time_list.append(torch.full((x.shape[0],), time, dtype=torch.long))
    Y_list = []
    for y, _, _ in layer.collected_Ys:
        Y_list.append(y)
    # concat input and output
    X = torch.cat(X_list, dim=0)
    Y = torch.cat(Y_list, dim=0)
    chunk_ids = torch.cat(chunk_ids_list, dim=0)
    time_ids = torch.cat(time_list, dim=0)


    X = X.to(device)
    Y = Y.to(device).float()

    # ========== calculate chunkwise weigts ==========
    raw_chunk_weights = layer.sensitivity if layer.sensitivity is not None else [1,1,1,1,1,1,1]
    mean_raw = sum(raw_chunk_weights) / len(raw_chunk_weights)
    scaled_chunk_weights = [w / mean_raw for w in raw_chunk_weights]
    weight_tensor_chunk = torch.tensor(scaled_chunk_weights, device=device)
    sample_weights_chunk = weight_tensor_chunk[chunk_ids].to(device)  # shape: [N]

    sample_weights = sample_weights_chunk

    if layer.use_dual_scale:
        sample_weights = torch.ones_like(sample_weights)
    logger.info(f"sample weight:{sample_weights}")

    n_total_item = X.size(0)
    for i in range(iters):
    
        idx = torch.randperm(n_total_item)[:batch_size]
        X_batch = X[idx]
        Y_batch = Y[idx]
        Y_quant_batch = layer(X_batch).float()
        weight_batch = sample_weights[idx]

        optimizer_w.zero_grad()
        optimizer_a.zero_grad()
        
        err = loss_func(Y_batch, Y_quant_batch, sample_weight=weight_batch)
        err.backward()

        optimizer_w.step()
        optimizer_a.step()
        
        scheduler_a.step()

    torch.cuda.empty_cache()
 
    layer.collected_Xs = []
    layer.collected_Ys = []

    # Finish optimization, use hard rounding; and merge alpha into weight
    def merge_alpha(weight, quant):
        w_floor = torch.floor(weight / quant.delta)
        w_int = w_floor + (quant.alpha >= 0).float()
        w_quant = torch.clamp(w_int + quant.zero_point, 0, quant.n_levels - 1)
        return (w_quant - quant.zero_point) * quant.delta
    
    if not layer.use_dual_scale:
        layer.weight_quantizer.soft_targets = False
        layer.weight.data.copy_(merge_alpha(layer.weight, layer.weight_quantizer))
        layer.weight_quantizer.alpha = None
    else:
        layer.weight_quantizer_outlier.soft_targets = False
        layer.weight_quantizer_normal.soft_targets = False
        for weight, quant in zip([layer.W_normal, layer.W_outlier], [layer.weight_quantizer_normal, layer.weight_quantizer_outlier]):
            weight.data.copy_(merge_alpha(weight, quant))
            quant.alpha = None
    

    # Reset original activation function:
    if not include_act_func:
        layer.activation_function = org_act_func


class LossFunction:
    def __init__(self,
                 layer: QuantModule,
                 round_loss: str = 'relaxation',
                 weight: float = 1.,
                 rec_loss: str = 'mse',
                 max_count: int = 2000,
                 b_range: tuple = (10, 2),
                 decay_start: float = 0.0,
                 warmup: float = 0.0,
                 p: float = 2.):

        self.layer = layer
        self.round_loss = round_loss
        self.weight = weight
        self.rec_loss = rec_loss
        self.loss_start = max_count * warmup
        self.p = p

        self.temp_decay = LinearTempDecay(max_count, rel_start_decay=warmup + (1 - warmup) * decay_start,
                                          start_b=b_range[0], end_b=b_range[1])
        self.count = 0

    def __call__(self, pred, tgt, grad=None, sample_weight=None):
        """
        Compute the total loss for adaptive rounding:
        rec_loss is the quadratic output reconstruction loss, round_loss is
        a regularization term to optimize the rounding policy

        :param pred: output from quantized model
        :param tgt: output from FP model
        :param grad: gradients to compute fisher information
        :return: total loss function
        """
        self.count += 1
        if self.rec_loss == 'mse':

            per_sample_loss = (pred - tgt).abs().pow(self.p).sum(1)      # 8x4680x1536 ->8x1536  
            sample_weight = sample_weight.view([-1] + [1] * (per_sample_loss.dim() - 1))   
            # ==========  chunkwise weigts ==========
            rec_loss = (per_sample_loss * sample_weight).mean()   

        elif self.rec_loss == 'fisher_diag':
            rec_loss = ((pred - tgt).pow(2) * grad.pow(2)).sum(1).mean()
        elif self.rec_loss == 'fisher_full':
            a = (pred - tgt).abs()
            grad = grad.abs()
            batch_dotprod = torch.sum(a * grad, (1, 2, 3)).view(-1, 1, 1, 1)
            rec_loss = (batch_dotprod * a * grad).mean() / 100
        elif self.rec_loss == 'cos':
            rec_loss = 1 - F.cosine_similarity(pred, tgt, dim=1).mean()
        else:
            raise ValueError('Not supported reconstruction loss function: {}'.format(self.rec_loss))

        b = self.temp_decay(self.count)
        if self.count < self.loss_start or self.round_loss == 'none':
            b = round_loss = 0
        elif self.round_loss == 'relaxation':
            round_loss = 0
            if not self.layer.use_dual_scale:
                round_vals = self.layer.weight_quantizer.get_soft_targets()
                round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
            else:
                for quant in [self.layer.weight_quantizer_normal, self.layer.weight_quantizer_outlier]:
                    round_vals = quant.get_soft_targets()
                    round_loss += self.weight * (1 - ((round_vals - .5).abs() * 2).pow(b)).sum()
        else:
            raise NotImplementedError

        total_loss = rec_loss + round_loss
        if self.count % 100 == 0:
            logger.info('Total loss:\t{:.3f} (rec:{:.3f}, round:{:.3f})\tb={:.2f}\tcount={}'.format(
                  float(total_loss), float(rec_loss), float(round_loss), b, self.count))
        return total_loss
    
class LinearTempDecay:
    def __init__(self, t_max: int, rel_start_decay: float = 0.2, start_b: int = 10, end_b: int = 2):
        self.t_max = t_max
        self.start_decay = rel_start_decay * t_max
        self.start_b = start_b
        self.end_b = end_b

    def __call__(self, t):
        """
        Cosine annealing scheduler for temperature b.
        :param t: the current time step
        :return: scheduled temperature
        """
        if t < self.start_decay:
            return self.start_b
        else:
            rel_t = (t - self.start_decay) / (self.t_max - self.start_decay)
            return self.end_b + (self.start_b - self.end_b) * max(0.0, (1 - rel_t))

