#---------------------------------------------------------------------------------#
# UniAD: Planning-oriented Autonomous Driving (https://arxiv.org/abs/2212.10156)  #
# Source code: https://github.com/OpenDriveLab/UniAD                              #
# Copyright (c) OpenDriveLab. All rights reserved.                                #
#---------------------------------------------------------------------------------#

"""
Custom collate function for temporal batch processing
Supports batch_size > 1 with temporal queue

Strategy: Flatten all nested lists but record batch boundaries in img_metas.
The model can use batch_size and queue_length to reconstruct batch structure.

IMPORTANT: mmcv scatter takes list[gpu_idx] for cpu_only DataContainer.
So we wrap all data in an outer list: [data] so scatter picks [0] = data.
"""

import torch
import numpy as np
from mmcv.parallel import DataContainer as DC


def _get_data(item):
    """Safely extract data from item, handling both DC and raw data."""
    if isinstance(item, DC):
        return item.data
    else:
        return item


def temporal_batch_collate(batch, samples_per_gpu=1):
    """
    Collate multiple temporal samples into a batch.
    
    Key design: 
    1. All nested GT lists are flattened into single lists
    2. Batch boundaries recorded in img_metas for reconstruction
    3. cpu_only data wrapped in outer list for scatter compatibility
       (scatter does list[gpu_idx], so [data] -> data after scatter)
    
    Args:
        batch: List[Dict], length = batch_size
    
    Returns:
        Dict with:
        - img: [B, L, N, C, H, W] stacked tensor
        - img_metas: wrapped List[List[Dict]] -> becomes List[Dict] after scatter
        - Temporal GT fields as wrapped [List] -> becomes List after scatter
        - Per-sample GT fields as wrapped [List] -> becomes List after scatter
    """
    batch_size = len(batch)
    
    if batch_size == 0:
        return {}
    
    # Get queue_length from first sample
    first_img = _get_data(batch[0]['img'])
    queue_length = first_img.shape[0]  # [L, N, C, H, W]
    
    # 1. Stack images: [B, L, N, C, H, W]
    imgs = torch.stack([_get_data(item['img']) for item in batch], dim=0)
    
    # 2. Collect img_metas and embed batch info
    # Flatten to list and record batch boundaries
    img_metas_list = []
    for b, item in enumerate(batch):
        sample_metas = _get_data(item['img_metas'])
        for t in range(queue_length):
            meta = sample_metas[t].copy()
            meta['batch_idx'] = b
            meta['temporal_idx'] = t
            img_metas_list.append(meta)
    
    # Add batch info to first meta for easy access
    img_metas_list[0]['batch_size'] = batch_size
    img_metas_list[0]['queue_length'] = queue_length
    
    # 3. Flatten temporal GT annotations: List[List[...]] -> List[...]
    # Length becomes batch_size * queue_length
    gt_bboxes_3d = []
    gt_labels_3d = []
    gt_inds = []
    gt_past_traj = []
    gt_past_traj_mask = []
    gt_sdc_bbox = []
    gt_sdc_label = []
    l2g_r_mat = []
    l2g_t = []
    timestamp = []
    
    for item in batch:
        gt_bboxes_3d.extend(_get_data(item['gt_bboxes_3d']))
        gt_labels_3d.extend(_get_data(item['gt_labels_3d']))
        gt_inds.extend(_get_data(item['gt_inds']))
        gt_past_traj.extend(_get_data(item['gt_past_traj']))
        gt_past_traj_mask.extend(_get_data(item['gt_past_traj_mask']))
        gt_sdc_bbox.extend(_get_data(item['gt_sdc_bbox']))
        gt_sdc_label.extend(_get_data(item['gt_sdc_label']))
        l2g_r_mat.extend(_get_data(item['l2g_r_mat']))
        l2g_t.extend(_get_data(item['l2g_t']))
        timestamp.extend(_get_data(item['timestamp']))
    
    # 4. Per-sample annotations (one per sample, variable length)
    gt_fut_traj = [_get_data(item['gt_fut_traj']) for item in batch]
    gt_fut_traj_mask = [_get_data(item['gt_fut_traj_mask']) for item in batch]
    gt_sdc_fut_traj = [_get_data(item['gt_sdc_fut_traj']) for item in batch]
    gt_sdc_fut_traj_mask = [_get_data(item['gt_sdc_fut_traj_mask']) for item in batch]
    
    # 5. Map/Lane related (one per sample)
    gt_lane_labels = [_get_data(item['gt_lane_labels']) for item in batch]
    gt_lane_bboxes = [_get_data(item['gt_lane_bboxes']) for item in batch]
    gt_lane_masks = [_get_data(item['gt_lane_masks']) for item in batch]
    
    # 6. Occupancy related (one per sample)
    gt_segmentation = [_get_data(item['gt_segmentation']) for item in batch]
    gt_instance = [_get_data(item['gt_instance']) for item in batch]
    gt_occ_img_is_valid = [_get_data(item['gt_occ_img_is_valid']) for item in batch]
    
    # Optional fields
    gt_centerness = None
    gt_offset = None
    gt_flow = None
    gt_backward_flow = None
    gt_occ_has_invalid_frame = None
    gt_future_boxes = None
    gt_future_labels = None
    sdc_planning = None
    sdc_planning_mask = None
    command = None
    
    if 'gt_centerness' in batch[0]:
        gt_centerness = [_get_data(item['gt_centerness']) for item in batch]
    if 'gt_offset' in batch[0]:
        gt_offset = [_get_data(item['gt_offset']) for item in batch]
    if 'gt_flow' in batch[0]:
        gt_flow = [_get_data(item['gt_flow']) for item in batch]
    if 'gt_backward_flow' in batch[0]:
        gt_backward_flow = [_get_data(item['gt_backward_flow']) for item in batch]
    if 'gt_occ_has_invalid_frame' in batch[0]:
        gt_occ_has_invalid_frame = [_get_data(item['gt_occ_has_invalid_frame']) for item in batch]
    if 'gt_future_boxes' in batch[0]:
        gt_future_boxes = [_get_data(item['gt_future_boxes']) for item in batch]
    if 'gt_future_labels' in batch[0]:
        gt_future_labels = [_get_data(item['gt_future_labels']) for item in batch]
    if 'sdc_planning' in batch[0]:
        sdc_planning = [_get_data(item['sdc_planning']) for item in batch]
    if 'sdc_planning_mask' in batch[0]:
        sdc_planning_mask = [_get_data(item['sdc_planning_mask']) for item in batch]
    if 'command' in batch[0]:
        command = [_get_data(item['command']) for item in batch]
    
    # Build output dict
    # CRITICAL: For cpu_only=True, scatter does list[gpu_idx]
    # So we wrap all data in [data] so after scatter it becomes data
    # This allows single-GPU to receive the full batch
    result = {
        'img': DC(imgs, cpu_only=False, stack=True),
        # Wrap in outer list for scatter compatibility
        'img_metas': DC([img_metas_list], cpu_only=True),
        # Flattened temporal GT (len = B * L), wrapped in outer list
        'gt_bboxes_3d': DC([gt_bboxes_3d], cpu_only=True),
        'gt_labels_3d': DC([gt_labels_3d], cpu_only=True),
        'gt_inds': DC([gt_inds], cpu_only=True),
        'gt_past_traj': DC([gt_past_traj], cpu_only=True),
        'gt_past_traj_mask': DC([gt_past_traj_mask], cpu_only=True),
        'gt_sdc_bbox': DC([gt_sdc_bbox], cpu_only=True),
        'gt_sdc_label': DC([gt_sdc_label], cpu_only=True),
        'l2g_r_mat': DC([l2g_r_mat], cpu_only=True),
        'l2g_t': DC([l2g_t], cpu_only=True),
        'timestamp': DC([timestamp], cpu_only=True),
        # Per-sample GT (len = B), wrapped in outer list
        'gt_fut_traj': DC([gt_fut_traj], cpu_only=True),
        'gt_fut_traj_mask': DC([gt_fut_traj_mask], cpu_only=True),
        'gt_sdc_fut_traj': DC([gt_sdc_fut_traj], cpu_only=True),
        'gt_sdc_fut_traj_mask': DC([gt_sdc_fut_traj_mask], cpu_only=True),
        'gt_lane_labels': DC([gt_lane_labels], cpu_only=True),
        'gt_lane_bboxes': DC([gt_lane_bboxes], cpu_only=True),
        'gt_lane_masks': DC([gt_lane_masks], cpu_only=True),
        'gt_segmentation': DC([gt_segmentation], cpu_only=True),
        'gt_instance': DC([gt_instance], cpu_only=True),
        'gt_occ_img_is_valid': DC([gt_occ_img_is_valid], cpu_only=True),
    }
    
    # Add optional fields (also wrapped in outer list)
    if gt_centerness is not None:
        result['gt_centerness'] = DC([gt_centerness], cpu_only=True)
    if gt_offset is not None:
        result['gt_offset'] = DC([gt_offset], cpu_only=True)
    if gt_flow is not None:
        result['gt_flow'] = DC([gt_flow], cpu_only=True)
    if gt_backward_flow is not None:
        result['gt_backward_flow'] = DC([gt_backward_flow], cpu_only=True)
    if gt_occ_has_invalid_frame is not None:
        result['gt_occ_has_invalid_frame'] = DC([gt_occ_has_invalid_frame], cpu_only=True)
    if gt_future_boxes is not None:
        result['gt_future_boxes'] = DC([gt_future_boxes], cpu_only=True)
    if gt_future_labels is not None:
        result['gt_future_labels'] = DC([gt_future_labels], cpu_only=True)
    if sdc_planning is not None:
        result['sdc_planning'] = DC([sdc_planning], cpu_only=True)
    if sdc_planning_mask is not None:
        result['sdc_planning_mask'] = DC([sdc_planning_mask], cpu_only=True)
    if command is not None:
        result['command'] = DC([command], cpu_only=True)
    
    return result


def reconstruct_batch_gt(gt_list, batch_size, queue_length):
    """
    Reconstruct batch structure from flattened GT list.
    
    Args:
        gt_list: Flattened list of length batch_size * queue_length
        batch_size: Number of samples in batch
        queue_length: Number of temporal frames per sample
    
    Returns:
        List[List[...]]: Nested list [batch_size][queue_length]
    """
    result = []
    for b in range(batch_size):
        start = b * queue_length
        end = start + queue_length
        result.append(gt_list[start:end])
    return result
