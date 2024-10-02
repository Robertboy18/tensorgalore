# copy dependencies from transformers/optimization.py
import math
import warnings
from typing import Callable, Iterable, Tuple, Union, List

import torch
from torch import nn
from torch.optim import Optimizer

#from transformers.utils.versions import require_version
from torch.autograd.profiler import profile, record_function

from .galore_projector import GaLoreProjector
from .tensor_galore_projector import TensorGaLoreProjector
from .training_utils import get_scheduler


class AdamW(Optimizer):
    """
    Implements Adam algorithm with weight decay fix as introduced in [Decoupled Weight Decay
    Regularization](https://arxiv.org/abs/1711.05101).

    Parameters:
        params (`Iterable[nn.parameter.Parameter]`):
            Iterable of parameters to optimize or dictionaries defining parameter groups.
        lr (`float`, *optional*, defaults to 0.001):
            The learning rate to use.
        betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`):
            Adam's betas parameters (b1, b2).
        eps (`float`, *optional*, defaults to 1e-06):
            Adam's epsilon for numerical stability.
        weight_decay (`float`, *optional*, defaults to 0.0):
            Decoupled weight decay to apply.
        correct_bias (`bool`, *optional*, defaults to `True`):
            Whether or not to correct bias in Adam (for instance, in Bert TF repository they use `False`).
        matrix_only : bool, default True
            whether to use Tensor Galore or flatten tensors to matrices and use classic Galore
        activation_checkpoint: bool, default True
            whether to use activation checkpointing during projection
        no_deprecation_warning (`bool`, *optional*, defaults to `False`):
            A flag used to disable the deprecation warning (set to `True` to disable the warning).
            
    """

    def __init__(
        self,
        params: Iterable[nn.parameter.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-6,
        weight_decay: float = 0.0,
        correct_bias: bool = True,
        matrix_only: bool = True,
        activation_checkpoint: bool = True,
        no_deprecation_warning: bool = False,
        first_dim_rollup = 0,
        support_complex: bool=False,
        warm_restart: bool=True,
        run_name=None
    ):
        if not no_deprecation_warning:
            warnings.warn(
                "This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch"
                " implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this"
                " warning",
                FutureWarning,
            )
       # require_version("torch>=1.5.0")  # add_ with alpha
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr} - should be >= 0.0")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[0]} - should be in [0.0, 1.0)")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter: {betas[1]} - should be in [0.0, 1.0)")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps} - should be >= 0.0")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay, "correct_bias": correct_bias}
        super().__init__(params, defaults)
        self.matrix_only = matrix_only
        self.activation_checkpoint = activation_checkpoint
        
        assert first_dim_rollup <= 3, "Error: cannot roll up more than 3 dimensions for first matrix dim"
        self.first_dim_rollup = first_dim_rollup
        self.support_complex = support_complex
        self.warm_restart = warm_restart
        self.run_name = run_name

    @torch.no_grad()
    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.

        Arguments:
            closure (`Callable`, *optional*): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]
                
                if "step" not in state:
                    state["step"] = 0
                    
                if 'dim' not in group:
                    group['dim'] = 2
                # GaLore Projection
                if "rank" in group:
                    if "projector" not in state:
                        if group['dim'] <= 2 or self.matrix_only:
                            state["projector"] = GaLoreProjector(
                                group["rank"], 
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"], 
                                proj_type=group["proj_type"],
                                activation_checkpoint=self.activation_checkpoint,
                                support_complex=self.support_complex)
                        else:
                            state["projector"] = TensorGaLoreProjector(
                                group["rank"], 
                                update_proj_gap=group["update_proj_gap"],
                                scale=group["scale"], 
                                proj_type=group["proj_type"], 
                                activation_checkpoint=self.activation_checkpoint,
                                warm_restart=self.warm_restart)
                    
                    # track tensor shape pre-flatten
                    input_shape = grad.shape
                    #print(f"{input_shape=}")
                    if grad.ndim == 5: # if complex tensor is stored as 2 real tensors
                        grad = torch.view_as_complex(grad)
                    #print(f"{grad.shape=}")
                    if self.matrix_only or grad.ndim <= 2:
                        proj_input = grad.view(grad.shape[0],-1)
                        if grad.ndim > 2:                            
                            input_shape = grad.shape
                            first_dim_reshape_size = math.prod(grad.shape[:self.first_dim_rollup])
                            proj_input = grad.view(first_dim_reshape_size, -1)
                        else:
                            proj_input = grad.view(grad.shape[0],-1)
                    else:
                        proj_input = grad
                    with record_function("#### GRAD FORWARD PROJ ####"):
                        grad = state["projector"].project(proj_input, state["step"])
                        #print(f"proj {grad.shape=}")
                        if self.run_name:
                            msg = f"Orig shape= {input_shape}\n"
                            msg += f"Proj shape= {grad.shape}"
                            with open(f"./memstats/{self.run_name}_grad_size", "w") as f:
                                f.write(msg)
                            f.close()
                # State initialization
                if "exp_avg" not in state:
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(grad)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(grad)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                # In-place operations to update the averages at the same time
                exp_avg.mul_(beta1).add_(grad, alpha=(1.0 - beta1))
                if torch.is_complex(grad) and self.support_complex:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad.conj(), value=1.0 - beta2)
                else:
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]
                if group["correct_bias"]:  # No bias correction for Bert
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = step_size * math.sqrt(bias_correction2) / bias_correction1

                # compute norm gradient
                norm_grad = exp_avg / denom
                
                # GaLore Projection Back
                if "rank" in group:
                    with record_function("#### GRAD BACKWARD PROJ ####"):
                        norm_grad = state["projector"].project_back(norm_grad)
                    if norm_grad.shape != input_shape:
                        # put complex grads (d1 x d2 x .. dn) back into real (d1 x d2 x .. dn x 2)
                        if torch.is_complex(norm_grad) and not torch.is_complex(p):
                            print(f"viewing {norm_grad.shape} as real")
                            norm_grad = torch.view_as_real(norm_grad)
                            #print(f"{norm_grad.shape=}")
                        #print(f"{input_shape=}")
                        norm_grad = norm_grad.view(input_shape)

                p.add_(norm_grad, alpha=-step_size)

                # Just adding the square of the weights to the loss function is *not*
                # the correct way of using L2 regularization/weight decay with Adam,
                # since that will interact with the m and v parameters in strange ways.
                #
                # Instead we want to decay the weights in a manner that doesn't interact
                # with the m/v parameters. This is equivalent to adding the square
                # of the weights to the loss with plain (non-momentum) SGD.
                # Add weight decay at the end (fixed version)
                if group["weight_decay"] > 0.0:
                    p.add_(p, alpha=(-group["lr"] * group["weight_decay"]))

        return loss


    @classmethod
    def per_layer_weight_opt(cls, 
                             model: torch.nn.Module,
                             id_galore_params: list,
                             rank: Union[int, float, List[int]], 
                             update_proj_gap: int,
                             galore_scale: float,
                             warm_restart: bool,
                             activation_checkpointing: bool,
                             proj_type: str,
                             lr: float,
                             weight_decay: float,
                             matrix_only: bool,
                             first_dim_rollup: int,
                             scheduler_name: str,
                             gamma: float,
                             patience: int,
                             T_max: int,
                             step_size: int):
        '''
        Optimize weight gradients per parameter and discard gradients immediately after stepping
        Returns a dict of optimizers per parameter
        '''
        optimizer_dict = {}
        scheduler_dict = {}
        galore_params = []
        galore_params.extend(list(model.fno_blocks.convs.parameters()))
        print(galore_params[0].shape, galore_params[1].shape, galore_params[2].shape, galore_params[3].shape)
        # drop the first projection layer
        galore_params.pop(0)
        id_galore_params = [id(p) for p in galore_params]
        # make parameters without "rank" to another group
        regular_params = [p for p in model.parameters() if id(p) not in id_galore_params]

        for p in regular_params:
            if p.requires_grad:
                optimizer_dict[p] = AdamW([p], lr=lr, 
                                          weight_decay=weight_decay, 
                                          activation_checkpoint=activation_checkpointing, 
                                          warm_restart=warm_restart)
                scheduler_dict[p] = get_scheduler(
                    scheduler_name=scheduler_name,
                    optimizer=optimizer_dict[p],
                    gamma=gamma,
                    patience=patience,
                    T_max=T_max,
                    step_size=step_size)
                
        for p in galore_params:
            if p.requires_grad:       
                optimizer_dict[p] = AdamW([{'params': [p], 
                                            'rank': rank, 
                                            'dim': p.ndim,
                                            'update_proj_gap': update_proj_gap * 2, 
                                            'scale': galore_scale, 
                                            'proj_type': proj_type}], 
                                            lr=lr, 
                                            weight_decay=weight_decay,
                                            matrix_only=matrix_only,
                                            first_dim_rollup=first_dim_rollup,
                                            activation_checkpoint=False)
                scheduler_dict[p] = get_scheduler(
                    scheduler_name=scheduler_name,
                    optimizer=optimizer_dict[p],
                    gamma=gamma,
                    patience=patience,
                    T_max=T_max,
                    step_size=step_size)
                    
        # define a hook function to update the parameter p during the backward pass
        def optimizer_hook(p):
            if p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter
        for p in regular_params:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)
        for p in galore_params:
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        return optimizer_dict, scheduler_dict