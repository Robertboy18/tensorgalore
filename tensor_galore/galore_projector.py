import torch
from tensorly import tenalg
from torch.utils.checkpoint import checkpoint

class GaLoreProjector:
    def __init__(self, rank, verbose=False, update_proj_gap=200, scale=1.0, proj_type='std', activation_checkpoint=False, support_complex=False):
        self.rank = rank
        self.verbose = verbose
        self.update_proj_gap = update_proj_gap
        self.scale = scale
        self.ortho_matrix = None
        self.proj_type = proj_type
        self.activation_checkpointing = activation_checkpoint
        self.support_complex = support_complex

    def project(self, full_rank_grad, iter):
        
        if self.proj_type == 'std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = optional_checkpoint_matmul(full_rank_grad, self.ortho_matrix.t(), self.activation_checkpointing)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = optional_checkpoint_matmul(self.ortho_matrix.t(), full_rank_grad, self.activation_checkpointing)
        elif self.proj_type == 'reverse_std':
            if full_rank_grad.shape[0] >= full_rank_grad.shape[1]:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
                low_rank_grad = optional_checkpoint_matmul(self.ortho_matrix.t(), full_rank_grad, self.activation_checkpointing)
            else:
                if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                    self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
                low_rank_grad = optional_checkpoint_matmul(full_rank_grad, self.ortho_matrix.t(), self.activation_checkpointing)
        elif self.proj_type == 'right':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='right')
            low_rank_grad = optional_checkpoint_matmul(full_rank_grad, self.ortho_matrix.t(), self.activation_checkpointing)
        elif self.proj_type == 'left':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='left')
            low_rank_grad = optional_checkpoint_matmul(self.ortho_matrix.t(), full_rank_grad, self.activation_checkpointing)
        elif self.proj_type == 'full':
            if self.ortho_matrix is None or iter % self.update_proj_gap == 0:
                self.ortho_matrix = self.get_orthogonal_matrix(full_rank_grad, self.rank, type='full')
            a = optional_checkpoint_matmul(self.ortho_matrix[0].t(), full_rank_grad, self.activation_checkpointing)
            low_rank_grad = optional_checkpoint_matmul(a, self.ortho_matrix[1].t(), self.activation_checkpointing)

        return low_rank_grad

    def project_back(self, low_rank_grad):

        if self.proj_type == 'std':
            if low_rank_grad.shape[0] >= low_rank_grad.shape[1]:
                full_rank_grad = optional_checkpoint_matmul(low_rank_grad, self.ortho_matrix, self.activation_checkpointing)
            else:
                full_rank_grad = optional_checkpoint_matmul(self.ortho_matrix, low_rank_grad, self.activation_checkpointing)
        elif self.proj_type == 'reverse_std':
            if low_rank_grad.shape[0] <= low_rank_grad.shape[1]: # note this is different from std
                full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
            else:
                full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'right':
            full_rank_grad = torch.matmul(low_rank_grad, self.ortho_matrix)
        elif self.proj_type == 'left':
            full_rank_grad = torch.matmul(self.ortho_matrix, low_rank_grad)
        elif self.proj_type == 'full':
            full_rank_grad = torch.matmul(self.ortho_matrix[0], low_rank_grad) @ self.ortho_matrix[1]

        return full_rank_grad * self.scale
    
    # svd decomposition
    def get_orthogonal_matrix(self, weights, rank, type):
        module_params = weights
        if torch.is_complex(module_params.data) and self.support_complex:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.cfloat()
        elif module_params.data.dtype != torch.float:
            float_data = False
            original_type = module_params.data.dtype
            original_device = module_params.data.device
            matrix = module_params.data.float()
        else:
            float_data = True
            matrix = module_params.data

        #make the smaller matrix always to be orthogonal matrix
        if type=='right':
            full_n_params = matrix.shape[0] * matrix.shape[1]
            # solve n_params equation for float ranks
            if isinstance(rank, float):
                low_rank_params = int(rank * full_n_params)
                int_rank = int(low_rank_params / matrix.shape[0])
            else:
                int_rank = rank
            _, _, Vh = torch.linalg.svd(matrix, full_matrices=False)
            B = Vh[:int_rank, :]
            if not float_data:
                B = B.to(original_device).type(original_type)
            return B
        elif type=='left':
            full_n_params = matrix.shape[0] * matrix.shape[1]
            # solve n_params equation for float ranks
            if isinstance(rank, float):
                low_rank_params = int(rank * full_n_params)
                int_rank = int(low_rank_params / matrix.shape[1])
            else:
                int_rank = rank
            U, _, _ = torch.linalg.svd(matrix, full_matrices=False)
            A = U[:, :int_rank]
            if not float_data:
                A = A.to(original_device).type(original_type)
            return A
        elif type=='full':
            A = U[:, :rank]
            B = Vh[:rank, :]
            if not float_data:
                A = A.to(original_device).type(original_type)
                B = B.to(original_device).type(original_type)
            return [A, B]
        else:
            raise ValueError('type should be left, right or full')

def optional_checkpoint_matmul(a: torch.Tensor, b: torch.Tensor, activation_checkpoint=True):
    """optional_checkpoint_matmul performs torch.matmul and optionally performs
    activation checkpointing. Removed from code to modularize and remove redundant lines

    Parameters
    ----------
    a : torch.Tensor
        input 1 to matmul
    b : torch.Tensor
        input 2 to matmul
    computes torch.matmul(a,b)
    checkpoint : bool, optional
        whether to perform activation checkpointing, by default True
    """
    if activation_checkpoint:
        return checkpoint(torch.matmul, a, b)
    else:
        return torch.matmul(a, b)
