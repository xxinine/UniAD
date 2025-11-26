_base_ = './base_track_map.py'

bf16 = dict()

optimizer_config = dict(
    grad_clip=dict(max_norm=35, norm_type=2)
)
