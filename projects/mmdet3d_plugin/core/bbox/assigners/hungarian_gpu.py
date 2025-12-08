# GPU-accelerated Hungarian Algorithm implementations
# Provides multiple backends for optimal assignment problem

import torch
import numpy as np

# Try to import GPU backends
_BACKEND = None

try:
    import cupy as cp
    from cupyx.scipy.optimize import linear_sum_assignment as cupy_linear_sum_assignment
    _BACKEND = 'cupy'
except ImportError:
    pass

if _BACKEND is None:
    try:
        # Alternative: lap library with CUDA support
        import lap
        _BACKEND = 'lap'
    except ImportError:
        pass

# Fallback to scipy
try:
    from scipy.optimize import linear_sum_assignment as scipy_linear_sum_assignment
except ImportError:
    scipy_linear_sum_assignment = None


def get_backend():
    """Get current Hungarian algorithm backend."""
    return _BACKEND if _BACKEND else 'scipy'


def linear_sum_assignment_gpu(cost_matrix, maximize=False):
    """
    GPU-accelerated Hungarian algorithm for optimal assignment.
    
    Automatically selects the best available backend:
    1. cupy (GPU) - fastest for large matrices
    2. lap (CPU, optimized C) - fast for medium matrices  
    3. scipy (CPU, Python) - fallback
    
    Args:
        cost_matrix: torch.Tensor [N, M] - cost matrix on GPU or CPU
        maximize: bool - if True, find maximum assignment instead of minimum
    
    Returns:
        row_indices: torch.Tensor [K] - matched row indices
        col_indices: torch.Tensor [K] - matched column indices
    
    Note: K = min(N, M) for complete assignment
    """
    device = cost_matrix.device
    
    if maximize:
        cost_matrix = -cost_matrix
    
    if _BACKEND == 'cupy' and cost_matrix.is_cuda:
        # GPU path using cupy
        row_ind, col_ind = _hungarian_cupy(cost_matrix)
    elif _BACKEND == 'lap':
        # Optimized CPU path using lap
        row_ind, col_ind = _hungarian_lap(cost_matrix)
    else:
        # Fallback to scipy
        row_ind, col_ind = _hungarian_scipy(cost_matrix)
    
    return row_ind.to(device), col_ind.to(device)


def _hungarian_cupy(cost_matrix):
    """Hungarian algorithm using cupy on GPU."""
    import cupy as cp
    from cupyx.scipy.optimize import linear_sum_assignment
    
    # Convert torch tensor to cupy array (zero-copy if on same GPU)
    if cost_matrix.is_cuda:
        # Direct memory sharing between PyTorch and CuPy
        cost_cp = cp.asarray(cost_matrix.detach())
    else:
        cost_cp = cp.asarray(cost_matrix.detach().numpy())
    
    # Run Hungarian on GPU
    row_ind, col_ind = linear_sum_assignment(cost_cp)
    
    # Convert back to torch
    row_ind = torch.as_tensor(cp.asnumpy(row_ind), dtype=torch.long)
    col_ind = torch.as_tensor(cp.asnumpy(col_ind), dtype=torch.long)
    
    return row_ind, col_ind


def _hungarian_lap(cost_matrix):
    """Hungarian algorithm using lap library (optimized C implementation)."""
    import lap
    
    cost_np = cost_matrix.detach().cpu().numpy().astype(np.float64)
    
    # lap.lapjv returns: cost, x, y
    # x[i] = j means row i is assigned to column j
    # y[j] = i means column j is assigned to row i
    _, x, _ = lap.lapjv(cost_np, extend_cost=True)
    
    # Convert to matched pairs
    row_ind = np.where(x >= 0)[0]
    col_ind = x[row_ind]
    
    row_ind = torch.from_numpy(row_ind.astype(np.int64))
    col_ind = torch.from_numpy(col_ind.astype(np.int64))
    
    return row_ind, col_ind


def _hungarian_scipy(cost_matrix):
    """Hungarian algorithm using scipy (fallback)."""
    if scipy_linear_sum_assignment is None:
        raise ImportError('Please run "pip install scipy" to install scipy first.')
    
    cost_np = cost_matrix.detach().cpu().numpy()
    row_ind, col_ind = scipy_linear_sum_assignment(cost_np)
    
    row_ind = torch.from_numpy(row_ind.astype(np.int64))
    col_ind = torch.from_numpy(col_ind.astype(np.int64))
    
    return row_ind, col_ind


def batched_linear_sum_assignment_gpu(cost_matrices, maximize=False):
    """
    Batched Hungarian algorithm for multiple cost matrices.
    
    This can be more efficient when processing multiple samples,
    as it avoids repeated Python overhead.
    
    Args:
        cost_matrices: List[torch.Tensor] or torch.Tensor [B, N, M]
        maximize: bool
    
    Returns:
        List of (row_indices, col_indices) tuples
    """
    if isinstance(cost_matrices, torch.Tensor):
        cost_matrices = [cost_matrices[i] for i in range(cost_matrices.size(0))]
    
    results = []
    for cost in cost_matrices:
        row_ind, col_ind = linear_sum_assignment_gpu(cost, maximize=maximize)
        results.append((row_ind, col_ind))
    
    return results


# Alias for drop-in replacement
linear_sum_assignment = linear_sum_assignment_gpu
