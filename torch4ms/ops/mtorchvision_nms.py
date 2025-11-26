"""
MindSpore implementation of Non-Maximum Suppression (NMS) for torchvision operations.
"""

import functools
from typing import List, Union, Optional, Tuple

import torch
import mindspore as ms
import mindspore.numpy as mnp
from mindspore import ops
from . import ops_registry

_NMS_TILE_SIZE = 256


def _bbox_overlap(boxes, gt_boxes):
  """Find Bounding box overlap.

  Args:
    boxes: first set of bounding boxes
    gt_boxes: second set of boxes to compute IOU

  Returns:
    iou: Intersection over union matrix of all input bounding boxes
  """
  bb_y_min, bb_x_min, bb_y_max, bb_x_max = ops.split(boxes, 4, axis=2)
  gt_y_min, gt_x_min, gt_y_max, gt_x_max = ops.split(gt_boxes, 4, axis=2)

  # Calculates the intersection area.
  i_xmin = ops.maximum(bb_x_min, ops.transpose(gt_x_min, (0, 2, 1)))
  i_xmax = ops.minimum(bb_x_max, ops.transpose(gt_x_max, (0, 2, 1)))
  i_ymin = ops.maximum(bb_y_min, ops.transpose(gt_y_min, (0, 2, 1)))
  i_ymax = ops.minimum(bb_y_max, ops.transpose(gt_y_max, (0, 2, 1)))
  i_area = ops.maximum((i_xmax - i_xmin), 0) * ops.maximum((i_ymax - i_ymin), 0)

  # Calculates the union area.
  bb_area = (bb_y_max - bb_y_min) * (bb_x_max - bb_x_min)
  gt_area = (gt_y_max - gt_y_min) * (gt_x_max - gt_x_min)
  # Adds a small epsilon to avoid divide-by-zero.
  u_area = bb_area + ops.transpose(gt_area, (0, 2, 1)) - i_area + 1e-8

  # Calculates IoU.
  iou = i_area / u_area

  return iou


def _self_suppression(in_args):
  iou, _, iou_sum = in_args
  batch_size = iou.shape[0]
  can_suppress_others = ops.reshape(
      ops.max(iou, 1) <= 0.5, (batch_size, -1, 1)).astype(iou.dtype)
  iou_suppressed = ops.reshape(
      (ops.max(can_suppress_others * iou, 1) <= 0.5).astype(
          iou.dtype), (batch_size, -1, 1)) * iou
  iou_sum_new = ops.sum(iou_suppressed, (1, 2))
  return iou_suppressed, ops.any(iou_sum - iou_sum_new > 0.5), iou_sum_new


def _cross_suppression(in_args):
  boxes, box_slice, iou_threshold, inner_idx = in_args
  batch_size = boxes.shape[0]
  # 使用gather替代lax.dynamic_slice
  indices = inner_idx * _NMS_TILE_SIZE
  new_slice = ops.slice(boxes, (0, indices, 0), (batch_size, _NMS_TILE_SIZE, 4))
  iou = _bbox_overlap(new_slice, box_slice)
  ret_slice = ops.expand_dims((ops.all(iou < iou_threshold, 1)).astype(
      box_slice.dtype), 2) * box_slice
  return boxes, ret_slice, iou_threshold, inner_idx + 1


class SuppressionLoopBody(ms.nn.Cell):
    """循环体Cell，用于while循环"""
    def __init__(self):
        super(SuppressionLoopBody, self).__init__()
        self._NMS_TILE_SIZE = _NMS_TILE_SIZE
        self.slice = ops.Slice()
        self.bbox_overlap = _bbox_overlap
        self.reshape = ops.Reshape()
        self.transpose = ops.Transpose()
        self.equal = ops.Equal()
        self.tile = ops.Tile()
        self.sum = ops.ReduceSum(keep_dims=False)
        self.any = ops.ReduceAny(keep_dims=False)
        self.expand_dims = ops.ExpandDims()
        self.arange = mnp.arange
        self.cross_suppression = _cross_suppression
        self.self_suppression = _self_suppression

    def construct(self, in_args):
        """Process boxes in the range [idx*_NMS_TILE_SIZE, (idx+1)*_NMS_TILE_SIZE).

        Args:
            in_args: A tuple of arguments: boxes, iou_threshold, output_size, idx

        Returns:
            boxes: updated boxes.
            iou_threshold: pass down iou_threshold to the next iteration.
            output_size: the updated output_size.
            idx: the updated induction variable.
        """
        boxes, iou_threshold, output_size, idx = in_args
        num_tiles = boxes.shape[1] // self._NMS_TILE_SIZE
        batch_size = boxes.shape[0]

        # Iterates over tiles that can possibly suppress the current tile.
        box_slice = self.slice(boxes, (0, idx * self._NMS_TILE_SIZE, 0), 
                              (batch_size, self._NMS_TILE_SIZE, 4))

        # 实现_cross_suppression的循环
        inner_idx = 0
        while inner_idx < idx:
            boxes, box_slice, _, inner_idx = self.cross_suppression(
                (boxes, box_slice, iou_threshold, inner_idx))

        # Iterates over the current tile to compute self-suppression.
        iou = self.bbox_overlap(box_slice, box_slice)
        # 创建mask
        arange_tile = self.reshape(self.arange(self._NMS_TILE_SIZE), (1, -1))
        mask = arange_tile > self.reshape(self.arange(self._NMS_TILE_SIZE), (-1, 1))
        mask = self.expand_dims(mask, 0)
        iou *= ((mask) & (iou >= iou_threshold)).astype(iou.dtype)

        # 实现_self_suppression的循环
        loop_condition = True
        iou_sum = self.sum(iou, (1, 2))
        while loop_condition:
            suppressed_iou, loop_condition, iou_sum = self.self_suppression(
                (iou, loop_condition, iou_sum))

        suppressed_box = self.sum(suppressed_iou, 1) > 0
        box_slice *= self.expand_dims(1.0 - suppressed_box.astype(box_slice.dtype), 2)

        # Uses box_slice to update the input boxes.
        arange_tiles = self.arange(num_tiles)
        mask = self.equal(arange_tiles, idx).astype(boxes.dtype)
        mask = self.reshape(mask, (1, -1, 1, 1))
        
        # 更新boxes
        box_slice_expanded = self.expand_dims(box_slice, 1)
        box_slice_tiled = self.tile(box_slice_expanded, (1, num_tiles, 1, 1))
        boxes_reshaped = self.reshape(boxes, (batch_size, num_tiles, self._NMS_TILE_SIZE, 4))
        boxes = box_slice_tiled * mask + boxes_reshaped * (1 - mask)
        boxes = self.reshape(boxes, (batch_size, -1, 4))

        # Updates output_size.
        output_size += self.sum(self.any(box_slice > 0, 2).astype(ms.int32), 1)
        return boxes, iou_threshold, output_size, idx + 1


def non_max_suppression_padded(scores, boxes, max_output_size, iou_threshold):
    """A wrapper that handles non-maximum suppression.

    Assumption:
        * The boxes are sorted by scores unless the box is a dot (all coordinates
          are zero).
        * Boxes with higher scores can be used to suppress boxes with lower scores.

    Args:
        scores: a tensor with a shape of [batch_size, anchors].
        boxes: a tensor with a shape of [batch_size, anchors, 4].
        max_output_size: a scalar integer representing the maximum number
          of boxes to be selected by non max suppression.
        iou_threshold: a float representing the threshold for deciding whether boxes
          overlap too much with respect to IOU.
    Returns:
        nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
          dtype as input scores.
        nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
          same dtype as input boxes.
    """
    # 转换为MindSpore tensor
    if not isinstance(scores, ms.Tensor):
        scores = ms.Tensor(scores, dtype=ms.float32)
    if not isinstance(boxes, ms.Tensor):
        boxes = ms.Tensor(boxes, dtype=ms.float32)
    
    batch_size = boxes.shape[0]
    num_boxes = boxes.shape[1]
    
    # 计算需要padding的数量
    pad = int(mnp.ceil(float(num_boxes) / _NMS_TILE_SIZE)) * _NMS_TILE_SIZE - num_boxes
    
    # padding操作
    boxes = ops.pad(boxes.astype(ms.float32), ((0, 0), (0, pad), (0, 0)))
    scores = ops.pad(scores.astype(ms.float32), ((0, 0), (0, pad)))
    num_boxes += pad

    # 创建循环体
    loop_body = SuppressionLoopBody()
    
    # 初始化循环变量
    output_size = ops.zeros((batch_size,), ms.int32)
    idx = 0
    
    # 执行循环
    while ops.min(output_size) < max_output_size and idx < num_boxes // _NMS_TILE_SIZE:
        boxes, iou_threshold, output_size, idx = loop_body((boxes, iou_threshold, output_size, idx))
    
    # 计算选中的索引
    mask = ops.any(boxes > 0, 2).astype(ms.int32)
    arange_expanded = ops.expand_dims(mnp.arange(num_boxes, 0, -1), 0)
    values = mask * arange_expanded
    
    # 使用topk获取选中的框
    _, indices = ops.top_k(values, max_output_size)
    idx = num_boxes - indices.astype(ms.int32)
    idx = ops.minimum(idx, num_boxes - 1)
    
    # 计算扁平化索引
    batch_indices = ops.reshape(mnp.arange(batch_size) * num_boxes, (-1, 1))
    idx = ops.reshape(idx + batch_indices, (-1,))
    
    # 重塑结果
    boxes_flat = ops.reshape(boxes, (-1, 4))
    selected_boxes = ops.gather(boxes_flat, idx, 0)
    boxes_result = ops.reshape(selected_boxes, (batch_size, max_output_size, 4))
    
    # 根据output_size进行mask
    mask_indices = ops.reshape(mnp.arange(max_output_size), (1, -1, 1))
    mask_output = ops.reshape(output_size, (-1, 1, 1))
    boxes_mask = (mask_indices < mask_output).astype(boxes_result.dtype)
    boxes_result = boxes_result * boxes_mask
    
    # 处理scores
    scores_flat = ops.reshape(scores, (-1, 1))
    selected_scores = ops.gather(scores_flat, idx, 0)
    scores_result = ops.reshape(selected_scores, (batch_size, max_output_size))
    
    # 根据output_size进行mask
    scores_mask = (mask_indices[:, :, 0] < mask_output[:, 0, 0]).astype(scores_result.dtype)
    scores_result = scores_result * scores_mask
    
    return scores_result, boxes_result


def nms(boxes, scores, iou_threshold):
    """MindSpore implementation of torchvision.nms
    
    Args:
        boxes: torch.Tensor of shape [N, 4]
        scores: torch.Tensor of shape [N]
        iou_threshold: float
        
    Returns:
        torch.Tensor of indices
    """
    # 将PyTorch张量转换为NumPy数组，再转换为MindSpore张量
    boxes_np = boxes.cpu().numpy() if boxes.device.type != 'cpu' else boxes.numpy()
    scores_np = scores.cpu().numpy() if scores.device.type != 'cpu' else scores.numpy()
    
    max_output_size = boxes.shape[0]
    boxes_ms = ms.Tensor(boxes_np.reshape((1, *boxes_np.shape)), dtype=ms.float32)
    scores_ms = ms.Tensor(scores_np.reshape((1, *scores_np.shape)), dtype=ms.float32)
    
    # 调用NMS实现
    _, selected_boxes = non_max_suppression_padded(
        scores_ms, boxes_ms, max_output_size, iou_threshold)
    
    # 找出非零的框索引
    non_zero_indices = []
    for i in range(selected_boxes.shape[1]):
        box = selected_boxes[0, i]
        if float(box[0]) > 0 or float(box[1]) > 0 or float(box[2]) > 0 or float(box[3]) > 0:
            non_zero_indices.append(i)
    
    # 返回PyTorch张量
    return torch.tensor(non_zero_indices, device=boxes.device)


# 注册操作
try:
    import torch
    import torchvision
    ops_registry.register_torch_dispatch_op(torch.ops.torchvision.nms, nms)
except Exception:
    pass