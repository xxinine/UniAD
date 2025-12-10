"""
BF16 Optimizer Hook for UniAD Training

This hook implements BF16 (Brain Floating Point 16) mixed precision training
following PyTorch's AMP best practices.

Key differences from FP16:
1. No loss scaling needed (BF16 has larger dynamic range)
2. Uses torch.cuda.amp.autocast for forward pass with custom dtype control
3. Model parameters stay in FP32
4. Gradients computed in FP32
5. DCN operations use FP32 (not supported in BF16)
"""

import torch
from torch.cuda.amp import autocast, custom_fwd, custom_bwd
from mmcv.runner import HOOKS, OptimizerHook
from mmcv.runner.dist_utils import allreduce_grads
from mmcv.utils import _BatchNorm, TORCH_VERSION, digit_version


@HOOKS.register_module()
class Bf16OptimizerHook(OptimizerHook):
    """BF16 Optimizer Hook for mixed precision training with gradient accumulation support.
    
    This hook wraps the model's forward method with BF16 autocast,
    while keeping parameters and gradients in FP32.
    DCN operations are kept in FP32 as they don't support BF16.
    
    Args:
        grad_clip (dict, optional): Gradient clipping config.
        coalesce (bool): Whether to coalesce gradient communication.
        bucket_size_mb (int): Bucket size for gradient coalescing.
        distributed (bool): Whether using distributed training.
        cumulative_iters (int): Number of iterations to accumulate gradients before update.
            Default: 1 (no accumulation). Set to >1 to simulate larger batch size.
    """
    
    def __init__(self,
                 grad_clip=None,
                 coalesce=True,
                 bucket_size_mb=-1,
                 distributed=False,
                 cumulative_iters=1):
        super(Bf16OptimizerHook, self).__init__(grad_clip)
        self.coalesce = coalesce
        self.bucket_size_mb = bucket_size_mb
        self.distributed = distributed
        self.cumulative_iters = cumulative_iters
        self._original_forward = None
        self._patched_dcn = False
        
        # Gradient accumulation state
        self._inner_iter = 0  # Current iteration within accumulation window
        
    def before_run(self, runner):
        """Initialize BF16 training by wrapping model forward.
        
        This method wraps the model's forward pass with autocast
        to enable BF16 computation, while keeping model parameters in FP32.
        DCN operations are kept in FP32 as they don't support BF16.
        """
        runner.logger.info('='*50)
        runner.logger.info('Initializing BF16 Mixed Precision Training')
        if self.cumulative_iters > 1:
            runner.logger.info(f'✓ Gradient Accumulation enabled: {self.cumulative_iters} iterations')
        runner.logger.info('='*50)
        
        # Check BF16 support
        if not torch.cuda.is_bf16_supported():
            runner.logger.warning(
                'BF16 is not supported on this GPU. '
                'Training may be slow or fail. '
                'Consider using FP16 instead.'
            )
        else:
            runner.logger.info(f'✓ BF16 is supported on current GPU')
        
        # Get the actual model (unwrap DDP if needed)
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        
        runner.logger.info(f'Model type: {type(model).__name__}')
        runner.logger.info(f'Model parameters will stay in FP32')
        runner.logger.info(f'Forward pass will use BF16 autocast')
        runner.logger.info(f'DCN operations will use FP32 (BF16 not supported)')
        
        # Patch DCN modules to use FP32
        self._patch_dcn_modules(model, runner)
        
        # Wrap the forward method with autocast
        self._wrap_model_forward(model)
        
        runner.logger.info('BF16 initialization complete')
        runner.logger.info('='*50)
    
    def _patch_dcn_modules(self, model, runner):
        """Patch MMCV CUDA modules to disable autocast (use FP32).
        
        Many MMCV CUDA operations don't support BF16, including:
        - ModulatedDeformConv2d (DCN)
        - FocalLoss
        - MultiScaleDeformableAttention
        - Other custom CUDA kernels
        
        We patch these modules to run in FP32 even when autocast is enabled.
        """
        patched_modules = []
        
        # List of module types that don't support BF16
        unsupported_types = []
        
        try:
            from mmcv.ops import ModulatedDeformConv2d
            unsupported_types.append((ModulatedDeformConv2d, 'DCN'))
        except ImportError:
            pass
            
        try:
            from mmdet.models.losses import FocalLoss
            unsupported_types.append((FocalLoss, 'FocalLoss'))
        except ImportError:
            pass
            
        try:
            from mmcv.ops import MultiScaleDeformableAttention
            unsupported_types.append((MultiScaleDeformableAttention, 'MultiScaleDeformableAttention'))
        except ImportError:
            pass
        
        if not unsupported_types:
            runner.logger.warning('No MMCV module types found for patching')
            return
        
        for name, module in model.named_modules():
            for module_type, type_name in unsupported_types:
                if isinstance(module, module_type):
                    # Store original forward
                    original_forward = module.forward
                    
                    def make_fp32_forward(orig_forward):
                        def fp32_forward(*args, **kwargs):
                            # Disable autocast for this module
                            with autocast(enabled=False):
                                # Convert inputs to FP32
                                args = tuple(
                                    arg.float() if torch.is_tensor(arg) and arg.is_floating_point() 
                                    else arg for arg in args
                                )
                                kwargs = {
                                    k: v.float() if torch.is_tensor(v) and v.is_floating_point() 
                                    else v for k, v in kwargs.items()
                                }
                                result = orig_forward(*args, **kwargs)
                                # Keep result in FP32, autocast context will handle conversion if needed
                                return result
                        return fp32_forward
                    
                    # Replace forward method
                    module.forward = make_fp32_forward(original_forward)
                    patched_modules.append(f"{type_name}")
                    break  # Only patch once per module
        
        if patched_modules:
            # Count by type
            from collections import Counter
            type_counts = Counter(patched_modules)
            runner.logger.info(f'✓ Patched {len(patched_modules)} modules to use FP32:')
            for type_name, count in type_counts.items():
                runner.logger.info(f'  - {type_name}: {count} instances')
            self._patched_dcn = True
        else:
            runner.logger.info('No unsupported modules found')
    
    def _wrap_model_forward(self, model):
        """Wrap model's forward method with BF16 autocast.
        
        This ensures that forward computations use BF16 precision
        while maintaining FP32 for parameters and gradients.
        DCN modules are already patched to use FP32.
        """
        # Store original forward method
        original_forward = model.__class__.forward
        self._original_forward = original_forward
        
        def wrapped_forward(self, *args, **kwargs):
            """Forward with BF16 autocast."""
            with autocast(dtype=torch.bfloat16, enabled=True):
                return original_forward(self, *args, **kwargs)
        
        # Replace forward method
        model.__class__.forward = wrapped_forward
    
    def after_train_iter(self, runner):
        """Perform backward pass with gradient accumulation support.
        
        This method implements gradient accumulation:
        1. Accumulate gradients over multiple iterations
        2. Only update parameters every cumulative_iters iterations
        3. Scale loss by 1/cumulative_iters to maintain effective learning rate
        
        Gradient accumulation flow:
        - iter 1-3: backward() but no optimizer.step()
        - iter 4: backward() + clip_grad + optimizer.step() + zero_grad()
        """
        # Scale loss for gradient accumulation
        # This ensures the effective gradient magnitude stays consistent
        loss = runner.outputs['loss'] / self.cumulative_iters
        
        # Backward pass to compute gradients (accumulate)
        loss.backward()
        
        # Increment inner iteration counter
        self._inner_iter += 1
        
        # Only update parameters when accumulation window is complete
        if self._inner_iter % self.cumulative_iters == 0:
            # All-reduce gradients in distributed training
            if runner.world_size > 1:
                allreduce_grads(
                    runner.model.parameters(),
                    self.coalesce,
                    self.bucket_size_mb
                )
            
            # Gradient clipping (on accumulated gradients)
            if self.grad_clip is not None:
                grad_norm = self.clip_grads(runner.model.parameters())
                if grad_norm is not None:
                    # Log gradient norm
                    runner.log_buffer.update({'grad_norm': float(grad_norm)},
                                            runner.outputs['num_samples'])
            
            # Optimizer step (update parameters)
            runner.optimizer.step()
            
            # Zero gradients for next accumulation window
            runner.optimizer.zero_grad()
            
            # Reset inner iteration counter
            self._inner_iter = 0
    
    def clip_grads(self, params):
        """Clip gradients.
        
        Args:
            params: Model parameters.
            
        Returns:
            float: Total norm of gradients, or None if no gradients.
        """
        params = list(
            filter(lambda p: p.requires_grad and p.grad is not None, params))
        
        if len(params) > 0:
            return torch.nn.utils.clip_grad_norm_(
                params, 
                **self.grad_clip
            )
        return None
