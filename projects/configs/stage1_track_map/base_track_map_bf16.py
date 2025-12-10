_base_ = './base_track_map.py'

bf16 = dict()

# Gradient Accumulation: simulate larger batch size without increasing memory
# This improves gradient stability and training quality
optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2),
    cumulative_iters=4,
)
