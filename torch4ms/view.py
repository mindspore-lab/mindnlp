import torch
import torch.utils._pytree as torch_pytree
import mindspore as ms
import mindspore.ops as ops
import mindspore.numpy as mnp
from enum import Enum
from typing import Union, List, Tuple, Optional, Any, cast
from abc import ABC, abstractmethod


class ViewInfoType(Enum):
    INVALID = 0
    NARROW = 1
    NO_OP = 2
    PERMUTE = 3
    RESHAPE = 4
    RESIZE = 5
    SELECT = 6
    AS_STRIDED = 7
    DIAGONAL = 8


class ViewInfo(ABC):
    """
    Abstract base class for all view operations.
    Defines the interface for applying and updating view transformations.
    """

    def __init__(self, view_info_type: ViewInfoType = ViewInfoType.INVALID):
        self.view_info_type = view_info_type

    @abstractmethod
    def update_tensor(self, new_value: ms.Tensor, ms_tensor: ms.Tensor) -> ms.Tensor:
        """
        Apply this view transformation to a MindSpore tensor and update its value.
        """
        pass

    @abstractmethod
    def transform_tensor(self, ms_tensor: ms.Tensor) -> ms.Tensor:
        """
        Apply this view transformation to a MindSpore tensor.
        """
        pass

    @abstractmethod
    def calculate_output_shape(self, source: ms.Tensor) -> List[int]:
        """
        Calculate the resulting shape after applying this view.
        """
        pass


class NarrowInfo(ViewInfo):
    """
    Represents a slicing operation on a tensor.
    Handles operations like tensor[1:3, :, 2:5:2].
    """

    def __init__(self, slices: Union[slice, Tuple[slice, ...]]) -> None:
        super().__init__(ViewInfoType.NARROW)
        self.slices = slices

    def __eq__(self, other: object) -> bool:
        return isinstance(other, NarrowInfo) and self.slices == other.slices

    def transform_tensor(self, ms_tensor: ms.Tensor) -> ms.Tensor:
        return ms_tensor[self.slices]

    def update_tensor(self, new_value: ms.Tensor, ms_tensor: ms.Tensor) -> ms.Tensor:
        # MindSpore 没有 .at[...].set(...)，用 scatter 实现
        idx = self._compute_scatter_indices(ms_tensor.shape)
        return ops.tensor_scatter_update(ms_tensor, idx, new_value)

    def calculate_output_shape(self, source: ms.Tensor) -> List[int]:
        return list(source[self.slices].shape)

    # ------------------------------------------------------------------
    # 辅助：把 slice 转换成 scatter 所需的 (N, ndim) 索引
    # ------------------------------------------------------------------
    def _compute_scatter_indices(self, shape: Tuple[int, ...]) -> ms.Tensor:
        # 这里仅示例：对于简单连续切片，可生成 meshgrid 索引
        # 实际生产环境需完整支持 stride、ellipsis 等
        ranges = []
        for s, dim in zip(self.slices, shape):
            start, stop, step = s.indices(dim)
            ranges.append(list(range(start, stop, step)))
        grid = np.meshgrid(*ranges, indexing='ij')
        idx = np.stack([g.ravel() for g in grid], axis=-1)
        return ms.Tensor(idx, dtype=ms.int32)


class SelectInfo(ViewInfo):
    def __init__(self, dim: int = 0, start: int = 0, end: int = 0, stride: int = 0):
        super().__init__(ViewInfoType.SELECT)
        self.dim, self.start, self.end, self.stride = dim, start, end, stride

    def __eq__(self, other: object) -> bool:
        return isinstance(other, SelectInfo) and \
               (self.dim, self.start, self.end, self.stride) == \
               (other.dim, other.start, other.end, other.stride)

    def transform_tensor(self, ms_tensor: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError("SelectInfo.transform_tensor not implemented")

    def update_tensor(self, new_value: ms.Tensor, ms_tensor: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError("SelectInfo.update_tensor not implemented")

    def calculate_output_shape(self, source: ms.Tensor) -> List[int]:
        raise NotImplementedError("SelectInfo.calculate_output_shape not implemented")


class AsStridedInfo(ViewInfo):
    def __init__(self, stride: List[int], offset: int = 0):
        super().__init__(ViewInfoType.AS_STRIDED)
        self.stride, self.offset = stride, offset

    def __eq__(self, other: object) -> bool:
        return isinstance(other, AsStridedInfo) and \
               self.stride == other.stride and self.offset == other.offset

    def transform_tensor(self, ms_tensor: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError("AsStridedInfo.transform_tensor not implemented")

    def update_tensor(self, new_value: ms.Tensor, ms_tensor: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError("AsStridedInfo.update_tensor not implemented")

    def calculate_output_shape(self, source: ms.Tensor) -> List[int]:
        raise NotImplementedError("AsStridedInfo.calculate_output_shape not implemented")


class DiagonalInfo(ViewInfo):
    def __init__(self, offset: int = 0, dim1: int = 0, dim2: int = 1):
        super().__init__(ViewInfoType.DIAGONAL)
        self.offset, self.dim1, self.dim2 = offset, dim1, dim2

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiagonalInfo) and \
               (self.offset, self.dim1, self.dim2) == (other.offset, other.dim1, other.dim2)

    def transform_tensor(self, ms_tensor: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError("DiagonalInfo.transform_tensor not implemented")

    def update_tensor(self, new_value: ms.Tensor, ms_tensor: ms.Tensor) -> ms.Tensor:
        raise NotImplementedError("DiagonalInfo.update_tensor not implemented")

    def calculate_output_shape(self, source: ms.Tensor) -> List[int]:
        raise NotImplementedError("DiagonalInfo.calculate_output_shape not implemented")


class View(torch.Tensor):
    """
    A View is a reference to another Tensor or another View,
    with a transformation applied to it.
    """

    @staticmethod
    def __new__(cls,
                parent: Union["torch4ms.Tensor", "View"],
                view_info: ViewInfo,
                env: Any) -> "View":
        shape = view_info.calculate_output_shape(parent.ms())
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            device="meta",
            dtype=parent.dtype,
            requires_grad=False,
        )

    def __init__(self,
                 parent: Union["torch4ms.Tensor", "View"],
                 view_info: ViewInfo,
                 env: Any) -> None:
        super().__init__()
        self.parent = parent
        self.view_info = view_info
        self._env = env

    def get_transformation_chain(self) -> List[ViewInfo]:
        if isinstance(self.parent, View):
            transformations = self.parent.get_transformation_chain()
            transformations.append(self.view_info)
            return transformations
        return [self.view_info]

    __torch_function__ = torch._C._disabled_torch_function_impl

    def source_ms(self) -> ms.Tensor:
        """Return the source MindSpore tensor."""
        if isinstance(self.parent, View):
            return self.parent.source_ms()
        return self.parent.ms()

    def replace_source_ms(self, new_value: ms.Tensor) -> None:
        """Update the source tensor with new values."""
        if isinstance(self.parent, View):
            self.parent.replace_source_ms(new_value)
        else:
            assert new_value.shape == self.parent._elem.shape
            self.parent._elem = new_value

    def torch(self) -> "torch4ms.Tensor":
        from torch4ms.tensor import Tensor
        return Tensor(self.ms(), self._env)

    def update(
        self,
        new_values: Union[ms.Tensor, "View", "torch4ms.Tensor"],
        view_infos: Optional[List[ViewInfo]] = None,
    ) -> None:
        if view_infos is None:
            view_infos = self.get_transformation_chain()

        source = self.source_ms()

        from torch4ms.tensor import Tensor
        if isinstance(new_values, (View, Tensor)):
            new_values = new_values.ms()

        # 正向计算中间结果
        intermediates = [source]
        for vi in view_infos[:-1]:
            intermediates.append(vi.transform_tensor(intermediates[-1]))

        # 反向传播更新
        for vi, parent in zip(reversed(view_infos), reversed(intermediates)):
            new_values = vi.update_tensor(new_values, parent)

        self.replace_source_ms(new_values)

    @classmethod
    def __torch_dispatch__(cls,
                           func: Any,
                           types: Tuple[Any, ...],
                           args: Tuple[Any, ...] = (),
                           kwargs: Optional[dict] = None):
        raise AssertionError(
            'torch4ms Tensors can only do math within the torch4ms environment. '
            'Please wrap your code with `with torch4ms.default_env()` or '
            'call torch4ms.enable_globally() before.'
        )

    def create_sub_view(self, view_info: ViewInfo) -> "View":
        return View(self, view_info, self._env)

    def __str__(self) -> str:
        return f"View({self.torch()})"

    def ms(self) -> ms.Tensor:
        """Return a copy of the source tensor after transformations."""
        result = self.source_ms()
        for vi in self.get_transformation_chain():
            result = vi.transform_tensor(result)
        return result

    __repr__ = __str__

    # 以下属性/方法保持与 PyTorch 接口兼容
    def dim(self):
        return self.ndim

    @property
    def device(self):
        return torch.device("ms:0")

    @property
    def ndim(self):
        return len(self.shape)