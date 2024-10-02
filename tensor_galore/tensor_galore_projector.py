import torch
from tensorly.decomposition import tucker, partial_tucker
from tensorly.decomposition._tucker import tucker_to_tensor
from tensorly import tenalg 
from torch.utils.checkpoint import checkpoint
from torch.autograd.profiler import record_function

class TensorGaLoreProjector:
    def __init__(self, rank, 
                 verbose=False, 
                 update_proj_gap=200, 
                 scale=1.0, 
                 proj_type='std', 
                 activation_checkpoint=False, 
                 warm_restart=False):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.proj_tensor = None
        self.proj_type = proj_type
        self.default = False
        self.activation_checkpoint = activation_checkpoint
        self.warm_restart = warm_restart
        
    def project(self, full_rank_grad, iter):
        with record_function("### GALORE PROJECT_FORWARD"):
            if self.proj_tensor is None and iter % self.update_proj_gap == 0:                    
                self.proj_tensor = self.get_projection_tensor(full_rank_grad, self.rank) 
            if isinstance(self.proj_tensor, tuple): # drop legacy core tensor
                core, self.proj_tensor = self.proj_tensor 
                del core
            self.proj_tensor = [factor.to(full_rank_grad.device) for factor in self.proj_tensor]
            transformed_low_rank = self.transform(self.proj_tensor, full_rank_grad)
        return transformed_low_rank

    def project_back(self, low_rank_grad):
        with record_function("#### GALORE PROJECT_BACK"):
            full_rank_grad = self.inverse_transform(self.proj_tensor, low_rank_grad)     
            return full_rank_grad * self.scale
            
    # Tucker decomp: higher-order SVD
    def get_projection_tensor(self, weights, rank):
        if torch.is_complex(weights.data) and weights.data.dtype != torch.cfloat:
            orig_dtype = weights.data.dtype
            matrix = weights.data.cfloat()
        else:
            matrix = weights.data

        # if warm_restart is True, initialize with 
        # existing projection tensor if it exists
        if self.warm_restart and self.proj_tensor is not None:
            init = self.proj_tensor
        else:
            init = 'svd' # default setting
        if self.activation_checkpoint:
            core, factors = checkpoint(tucker, matrix, rank=rank, init=init)
        else:
            core, factors = tucker(matrix, rank=rank, init=init)
        del core
        return factors

    def transform(self, factors, x):
        with record_function("### TENSOR_GALORE_TRANSFORM"):
            # unpack/drop core
            if self.activation_checkpoint:
                return checkpoint(tenalg.multi_mode_dot, x, factors, Transpose=True)
            return tenalg.multi_mode_dot(x, factors, transpose=True)

    def inverse_transform(self, factors, x):
        with record_function("### TENSOR_GALORE_INV_TRANSFORM"):
            if self.activation_checkpoint:
                return checkpoint(tenalg.multi_mode_dot, x, factors)
            return tenalg.multi_mode_dot(x, factors)
            

