import threading
import logging
import sys
import contextlib
from typing import Optional, Any
import torch._dispatch
# no_dispatch在torch._dispatch模块中
import mindspore
import mindspore.numpy as mnp
import numpy as np
import itertools
import torch
import torch.utils._pytree as torch_pytree
import torch.utils._mode_utils as mode_utils
import torch.utils._python_dispatch as torch_dispatch
from mindspore import Tensor as ms_Tensor
from mindspore import Parameter
from mindspore import nn
from mindspore import ops
from torch4ms.view import View
from torch4ms import config
from torch4ms.ops import mappings, ops_registry
from torch4ms import amp

logger = logging.getLogger(__name__)


class OperatorNotFound(Exception):
    """
    当找不到对应的算子实现时抛出的异常

    这个异常用于在torch4ms无法找到某个PyTorch操作对应的MindSpore实现时抛出，
    提示用户该操作可能尚未支持或需要自定义实现。
    """
    pass


@contextlib.contextmanager
def log_nested(env, message):
    """
    嵌套操作日志上下文管理器

    用于打印带有缩进的嵌套日志信息，帮助调试复杂的操作转换过程。
    仅当debug_print_each_op配置为True时才会输出日志。

    Args:
        env: Environment实例，包含配置信息
        message: 要打印的日志消息
    """
    # 仅在调试模式下打印日志
    if env.config.debug_print_each_op:
        print((" " * log_nested.level) + message, file=sys.stderr)
    # 增加缩进级别
    log_nested.level += 1
    yield
    # 恢复缩进级别
    log_nested.level -= 1

# 缩进级别初始化
log_nested.level = 0

class Tensor(torch.Tensor):
    """
  torch4ms的核心张量类，封装MindSpore的Tensor

  该类封装了MindSpore张量，并提供了与原始接口兼容的方法，使得操作能够透明地转换为MindSpore执行。
  参考 torchax 的实现，继承自 torch.Tensor 以通过 PyTorch 的类型检查。
    """

    @staticmethod
    def __new__(cls, elem, env, requires_grad=False):
        """
        创建新的Tensor实例，使用 torch.Tensor._make_wrapper_subclass

        Args:
            elem: MindSpore张量或可以转换为MindSpore张量的数据
            env: Environment对象，包含转换环境
            requires_grad: 是否需要梯度计算

        Returns:
            新的Tensor实例
        """
        # 确保 elem 是 MindSpore Tensor
        if not isinstance(elem, ms_Tensor):
            elem = ms_Tensor(elem)
        
        # 获取形状和类型信息
        shape = elem.shape
        dtype = elem.dtype
        
        # 使用 torch.Tensor._make_wrapper_subclass 创建子类实例（参考 torchax 和 View 的实现）
        # 将 MindSpore dtype 转换为 PyTorch dtype
        torch_dtype = mappings.ms2t_dtype(dtype)
        return torch.Tensor._make_wrapper_subclass(
            cls,
            shape,
            dtype=torch_dtype,
            device='meta',  # 使用 meta 设备，因为实际数据存储在 _elem 中
            requires_grad=requires_grad,
        )

    def __init__(self, elem, env, requires_grad=False):
        """
        初始化Tensor实例

        Args:
            elem: MindSpore张量或可以转换为MindSpore张量的数据
            env: Environment对象，包含转换环境
            requires_grad: 是否需要梯度计算
        """
        super().__init__()
        if not isinstance(elem, ms_Tensor):
            elem = ms_Tensor(elem)

        self._elem = elem
        self._env = env
        self._requires_grad = requires_grad

        if requires_grad:
            self._elem = Parameter(elem)

    def __str__(self):
        return f'torch4ms.Tensor({self._elem})'

    __repr__ = __str__

    @property
    def shape(self):
        return self._elem.shape

    @property
    def ndim(self):
        return self._elem.ndim

    def flatten(self, start_dim=0, end_dim=-1):
        # 使用MindSpore的reshape操作
        return self._env.ms2t_iso(mnp.reshape(self._elem, (-1,)))

    # ========= 基本算术运算支持 =========
    def _binary_op(self, other, ms_op):
        """
        通用二元运算封装，使用MindSpore算子在内部张量上计算。
        """
        # 解包对端张量 / 标量，尽量兼容 PyTorch / NumPy / Python 原生类型
        if isinstance(other, Tensor):
            other_elem = other._elem
        elif isinstance(other, ms_Tensor):
            other_elem = other
        elif isinstance(other, torch.Tensor):
            # 从 PyTorch Tensor 转成 MindSpore Tensor
            other_elem = ms_Tensor(other.detach().cpu().numpy())
        elif isinstance(other, np.ndarray):
            other_elem = ms_Tensor(other)
        else:
            # Python 标量或可转换对象
            other_elem = ms_Tensor(other)

        res = ms_op(self._elem, other_elem)
        return self._env.ms2t_iso(res)

    def __add__(self, other):
        """支持 x + y 形式的加法运算。"""
        return self._binary_op(other, ops.add)

    def __radd__(self, other):
        """支持 y + x 形式的加法运算。"""
        return self._binary_op(other, ops.add)

    def __sub__(self, other):
        """支持 x - y。"""
        return self._binary_op(other, ops.sub)

    def __rsub__(self, other):
        """支持 y - x。"""
        # 交换顺序：other - self == -(self - other)，这里直接用 MindSpore sub 反向计算
        if isinstance(other, Tensor):
            return other._binary_op(self, ops.sub)
        return self._binary_op(other, lambda a, b: ops.sub(b, a))

    def __mul__(self, other):
        """支持 x * y（逐元素乘法）。"""
        return self._binary_op(other, ops.mul)

    def __rmul__(self, other):
        """支持 y * x。"""
        return self._binary_op(other, ops.mul)

    def __truediv__(self, other):
        """支持 x / y。"""
        return self._binary_op(other, ops.div)

    def __rtruediv__(self, other):
        """支持 y / x。"""
        if isinstance(other, Tensor):
            return other._binary_op(self, ops.div)
        return self._binary_op(other, lambda a, b: ops.div(b, a))

    def __floordiv__(self, other):
        """支持 x // y。"""
        return self._binary_op(other, ops.floor_div)

    def __rfloordiv__(self, other):
        """支持 y // x。"""
        if isinstance(other, Tensor):
            return other._binary_op(self, ops.floor_div)
        return self._binary_op(other, lambda a, b: ops.floor_div(b, a))

    def __pow__(self, other):
        """支持 x ** y。"""
        return self._binary_op(other, ops.pow)

    def __rpow__(self, other):
        """支持 y ** x。"""
        if isinstance(other, Tensor):
            return other._binary_op(self, ops.pow)
        return self._binary_op(other, lambda a, b: ops.pow(b, a))

    # 比较运算，返回布尔 Tensor
    def __eq__(self, other):
        return self._binary_op(other, ops.equal)

    def __ne__(self, other):
        return self._binary_op(other, ops.not_equal)

    def __lt__(self, other):
        return self._binary_op(other, ops.less)

    def __le__(self, other):
        return self._binary_op(other, ops.less_equal)

    def __gt__(self, other):
        return self._binary_op(other, ops.greater)

    def __ge__(self, other):
        return self._binary_op(other, ops.greater_equal)

    def __setitem__(self, key, val):
        # 确保索引操作在内部张量上执行
        if isinstance(val, Tensor):
            val = val._elem
        self._elem[key] = val

    def type_as(self, other):
        # 确保other是Tensor类型
        if not isinstance(other, Tensor):
            raise TypeError(f"Expected Tensor, got {type(other).__name__}")
        # 获取目标数据类型并转换
        target_dtype = other.dtype
        return self._env.ms2t_iso(mnp.astype(self._elem, target_dtype))

    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        """
        PyTorch的分发机制入口点，捕获对张量的操作

        Args:
            func: 要执行的PyTorch函数
            types: 参数类型列表
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            操作结果
        """
        # 特殊处理wait_tensor操作
        if func == torch.ops._c10d_functional.wait_tensor.default:
            return args[0]._env.dispatch(func, types, args, kwargs)
        # 特殊处理device操作，返回privateuseone设备
        if func == torch.ops.prim.device.default:
            return torch.device('privateuseone', 0)
        # 确保在torch4ms环境中使用
        raise AssertionError(
            'torch4ms Tensors can only do math within the torch4ms environment.'
            'Please wrap your code with `with torch4ms.default_env()` or '
            'call torch4ms.enable_globally() before.')

    def detach(self):
        # MindSpore中使用stop_gradient方法
        detached_elem = ops.stop_gradient(self._elem)
        return Tensor(detached_elem, self._env, requires_grad=False)

    def numpy(self) -> np.ndarray:
        return self._elem.asnumpy()

    def mindspore(self) -> ms_Tensor:
        return self._elem
    
    def ms(self) -> ms_Tensor:
        """Alias for mindspore() for compatibility with View class."""
        return self._elem

    def torch(self) -> "Tensor":
        # 在MindSpore实现中，这将返回一个兼容的张量
        return self._env.ms2t_copy(self.mindspore())

    @property
    def dtype(self):
        return self._elem.dtype

    def dim(self):
        return self.ndim

    @property
    def device(self):
        # 返回MindSpore设备信息
        return str(self._elem.device)

    @property
    def data(self):
        logger.warning("In-place to .data modifications might have different behavior in MindSpore")
        return self

    @data.setter
    def data(self, other):
        if isinstance(other, Tensor):
            self._elem = other._elem
        elif isinstance(other, ms_Tensor):
            self._elem = other
        else:
            self._elem = ms_Tensor(other)

    def apply_mindspore(self, ms_function, *args, **kwargs):
        """
        在内部MindSpore张量上应用MindSpore函数

        Args:
            ms_function: 要应用的MindSpore函数
            *args: 传递给MindSpore函数的位置参数
            **kwargs: 传递给MindSpore函数的关键字参数

        Returns:
            转换后的torch4ms.Tensor结果
        """
        # 在内部MindSpore张量上调用函数
        res = ms_function(self._elem, *args, **kwargs)
        # 将结果转换回torch4ms.Tensor
        return self._env.ms2t_iso(res)

    def apply_mindspore_(self, ms_function, *args, **kwargs):
        self._elem = ms_function(self._elem, *args, **kwargs)
        return self

    def tolist(self):
        return self._elem.asnumpy().tolist()


def debug_accuracy(func, args, kwargs, current_output):
    """
    调试PyTorch和MindSpore结果的精度差异

    比较PyTorch原生执行和torch4ms执行结果之间的数值差异，用于验证转换的准确性。

    Args:
        func: PyTorch函数
        args: 函数参数
        kwargs: 关键字参数
        current_output: torch4ms执行的结果

    Returns:
        bool: 如果结果在容限范围内则返回True，否则返回False
    """
    args_torch, kwargs_torch, out_torch = torch_pytree.tree_map_only(
        torch.Tensor, lambda x: x.torch(), (args, kwargs, current_output))

    with torch._C.DisableTorchFunction():
        if "device" in kwargs_torch:
            kwargs_torch["device"] = "cpu"  # do the torch native for comparison
        expected_out = func(*args_torch, **kwargs_torch)

    flattened_current_out, _ = torch_pytree.tree_flatten(out_torch)
    flattened_expected_out, _ = torch_pytree.tree_flatten(expected_out)

    for ex, real in zip(flattened_expected_out, flattened_current_out):
        if isinstance(ex, torch.Tensor) and ex.dtype != real.dtype:
            ex = ex.to(real.dtype)
        try:
            if isinstance(ex, torch.Tensor) and not torch.allclose(
                    ex, real, atol=1e-3, equal_nan=True):
                import pdb

                pdb.set_trace()
        except:
            import pdb

            pdb.set_trace()

    return True


def _make_debug_msg(is_dispatch, log_args, func, args, kwargs):
    """
    生成调试消息字符串

    创建格式化的调试信息，包含函数名称、参数类型和形状等信息。

    Args:
        is_dispatch: 是否为分发模式
        log_args: 是否记录参数详情
        func: 函数对象
        args: 函数参数
        kwargs: 关键字参数

    Returns:
        str: 格式化的调试消息
    """

    def _display(a):
        if isinstance(a, torch.Tensor):
            return f"Tensor of {type(a)}: {a.dtype}{a.shape}"
        elif isinstance(a, ms_Tensor):
            return f"MindSpore Tensor of {type(a)}: {a.dtype}{a.shape}"
        else:
            return str(a)

    kwargs = kwargs or {}
    title = "DISPATCH" if is_dispatch else "FUNCTION"
    args_msg = "args: " + ",".join(_display(a) for a in args) if log_args else ""
    kwargs_msg = ("kwargs: " +
                  ",".join(f"{key}: {_display(a)}" for key, a in kwargs.items())
                  if log_args else "")
    return f"{title}: {_name_of_func(func)} {args_msg} ~ {kwargs_msg}"


class XLAFunctionMode(torch.overrides.TorchFunctionMode):
    """
    torch4ms的PyTorch函数模式类

    用于拦截和处理PyTorch的函数调用，实现PyTorch函数到Mindspore函数的转换。
    继承自PyTorch的TorchFunctionMode以拦截全局函数调用。

    Args:
        env: Environment实例，负责实际的操作转换逻辑
    """

    def __init__(self, env):
        self.env = env

    def __torch_function__(self,
                           func,
                           types,
                           args=(),
                           kwargs=None) -> torch.Tensor:
        """
        PyTorch函数钩子，处理所有全局PyTorch函数调用

        Args:
            func: 要执行的PyTorch函数
            types: 参数类型列表
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            转换后的计算结果
        """
        message = f"FUNCTION: {_name_of_func(func)}"
        if self.env.config.debug_print_each_op_operands:
            message = message + "f"
        message = _make_debug_msg(False,
                                  self.env.config.debug_print_each_op_operands,
                                  func, args, kwargs)
        with log_nested(self.env, message):
            try:
                return self.env.dispatch(func, types, args, kwargs)
            except OperatorNotFound:
                pass
            if _name_of_func(func) in (
                    "rot90"):  # skip rot90 with k%4==0 due to no change
                if len(args) >= 2 and type(args[1]) == int:
                    if (args[1]) % 4 == 0:
                        return args[0]
            # 如果找不到操作实现，尝试在 no_dispatch 上下文中调用原始函数
            # 这允许 PyTorch 原生函数处理 torch4ms.Tensor（如果支持）
            with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                try:
                    # 尝试将 torch4ms.Tensor 转换为普通 torch.Tensor（如果需要）
                    converted_args = []
                    for arg in args:
                        if isinstance(arg, Tensor):
                            converted_args.append(self.env.ms2t_copy(arg.mindspore()))
                        elif isinstance(arg, View):
                            converted_args.append(arg.torch())
                        else:
                            converted_args.append(arg)
                    converted_kwargs = {}
                    for k, v in (kwargs or {}).items():
                        if isinstance(v, Tensor):
                            converted_kwargs[k] = self.env.ms2t_copy(v.mindspore())
                        elif isinstance(v, View):
                            converted_kwargs[k] = v.torch()
                        else:
                            converted_kwargs[k] = v
                    result = func(*converted_args, **converted_kwargs)
                    # 如果结果需要转换回 torch4ms.Tensor，使用环境转换
                    return result
                except Exception:
                    # 如果转换失败，抛出 OperatorNotFound 让上层处理
                    raise OperatorNotFound(
                        f"Operator with name {_name_of_func(func)} has no lowering and cannot fallback to PyTorch native implementation"
                    )


class XLADispatchMode(torch_dispatch.TorchDispatchMode):
    """
    torch4ms的PyTorch分发模式类

    用于拦截和处理PyTorch的底层分发操作，实现PyTorch操作到Mindspore操作的转换。
    继承自PyTorch的TorchDispatchMode以拦截底层算子调用。

    Args:
        env: Environment实例，负责实际的操作转换逻辑
    """

    def __init__(self, env):
        self.env = env

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        """
        PyTorch分发钩子，处理所有的PyTorch底层操作

        Args:
            func: 要执行的PyTorch函数
            types: 参数类型列表
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            转换后的计算结果
        """
        message = _make_debug_msg(True,
                                  self.env.config.debug_print_each_op_operands,
                                  func, args, kwargs)
        with log_nested(self.env, message):
            if isinstance(func, torch._ops.OpOverloadPacket):
                with self:
                    return func(*args, **kwargs)
            # Only functions under these namespaces will be intercepted
            if func.namespace not in (
                    "aten",
                    "_c10d_functional",
                    "torchvision",
                    "xla",
            ):
                return func(*args, **kwargs)
            return self.env.dispatch(func, types, args, kwargs)

def _name_of_func(func_or_name):
    """
    获取函数的名称

    处理函数对象或名称字符串，返回规范化的函数名称。

    Args:
        func_or_name: 函数对象或字符串名称

    Returns:
        str: 函数的名称或字符串表示
    """
    if isinstance(func_or_name, str):
        return func_or_name
    
    # 处理PyTorch OpOverload对象
    if hasattr(func_or_name, "name"):
        # 先尝试直接获取name属性
        try:
            name = func_or_name.name
            # 检查是否是字符串
            if isinstance(name, str):
                return name
        except Exception:
            pass
    
    # 处理PyTorch OpOverload对象的另一种方式
    if hasattr(func_or_name, "__repr__"):
        # 获取字符串表示，然后解析出算子名称
        repr_str = repr(func_or_name)
        # 格式通常是：<OpOverload(op='aten.lift_fresh', overload='default')>
        if "op='" in repr_str:
            op_part = repr_str.split("op='")[1].split("'").pop(0)
            return op_part
    
    # 处理有__name__属性的对象
    if hasattr(func_or_name, "__name__"):
        return func_or_name.__name__
    
    # 作为最后的手段，返回字符串表示
    return str(func_or_name)


# 支持的张量构造函数集合 - MindSpore实现
# 这些是常见的张量创建函数，torch4ms会拦截这些函数并使用MindSpore实现
TENSOR_CONSTRUCTORS = {
    "ones",  # 创建全1张量
    "zeros",  # 创建全0张量
    "empty",  # 创建未初始化张量
    "tensor",  # 从数据创建张量
    "arange",  # 创建等差数列张量
    "eye",  # 创建单位矩阵
    "randn",  # 创建正态分布随机张量
    "rand",  # 创建均匀分布随机张量
    "randint",  # 创建整数随机张量
    "full",  # 创建填充指定值的张量
    "as_tensor",  # 转换为张量
}

# TODO(wen): use existing types, either from torch or mindspore
SUPPORTED_MINDSRORE_PLATFORM = ["cpu", "gpu", "npu"]


class RuntimeProperty:
    """
    运行时属性管理类

    管理torch4ms的运行时配置，包括设备网格、随机数生成器和自动混合精度数据类型等。

    Attributes:
        mesh: 设备网格配置，用于并行计算
        prng: PRNG随机数生成器密钥
        autocast_dtype: 自动混合精度使用的数据类型
    """
    mesh: Any
    prng: Any
    autocast_dtype: Any

    def __init__(self, mesh, prng, autocast_dtype):
        """
        初始化运行时属性

        Args:
            mesh: 设备网格配置
            prng: 初始PRNG密钥
            autocast_dtype: 自动混合精度数据类型
        """
        self.mesh = mesh
        self.prng = prng
        self.autocast_dtype = autocast_dtype

    def override(self, **kwargs):
        """
        创建属性覆盖对象

        创建一个新的OverrideProperty对象，用于临时覆盖当前属性。

        Args:
            **kwargs: 要覆盖的属性及其新值

        Returns:
            OverrideProperty: 覆盖属性对象
        """
        return OverrideProperty(self, kwargs)

    def get_and_rotate_prng_key(self):
        """
        获取并旋转PRNG密钥

        这是确保随机操作一致性的实现方式，每次调用此方法都会产生一个新的随机密钥，并更新内部状态。

        Returns:
            用于随机操作的新PRNG密钥
        """
        old_key = self.prng
        # 简单递增作为新种子，确保随机性
        self.prng = (old_key + 1) % (1 << 31)
        return old_key


class OverrideProperty(RuntimeProperty):
    """
    属性覆盖类

    允许临时覆盖运行时属性的类，当请求属性时，首先检查覆盖字典，
    如果不存在则回退到父属性对象。

    Args:
        parent: 父RuntimeProperty对象
        override: 要覆盖的属性字典
    """

    def __init__(self, parent, override):
        """
        初始化覆盖属性

        Args:
            parent: 父属性对象
            override: 覆盖属性字典
        """
        self.parent = parent
        self._override = dict(override)

    def __getattr__(self, name):
        """
        获取属性，优先使用覆盖值

        当请求属性时，首先检查是否有覆盖值，如果有则返回覆盖值，
        否则从父属性对象中获取。

        Args:
            name: 属性名称

        Returns:
            属性值
        """
        if name in self._override:
            return self._override[name]
        return getattr(self.parent, name)


class Environment(contextlib.ContextDecorator):
    """
   torch4ms的核心环境类，管理PyTorch到MindSpore的转换过程

   该类作为上下文管理器，负责：
   - 维护算子注册表和分解函数
   - 管理随机数生成
   - 配置项管理
   - 操作分发和执行
   - 张量转换
    """

    def __init__(self, configuration=None):
        """
        初始化Environment实例

        Args:
            configuration: 可选的配置对象，默认使用默认配置
        """
        # 初始化MindSpore处理类
        self._function_mode = XLAFunctionMode(self)
        self._dispatch_mode = XLADispatchMode(self)

        # 算子注册表：存储PyTorch到MindSpore的映射
        self._ops = {}
        # 分解函数表：存储复杂操作的分解实现
        self._decomps = {}

        # 加载注册的算子
        self.load_ops()

        # 初始化网格（分布式计算用）
        _mesh = None
        # 加载配置
        self.config = configuration or config.Configuration()

        # 根据配置设置MindSpore运行模式
        try:
            from mindspore import context
            
            # 安全获取配置属性，确保即使属性不存在也不会出错
            use_graph_mode = getattr(self.config, 'use_ms_graph_mode', False)
            device_target = getattr(self.config, 'default_device_target', 'CPU')
            
            # 打印调试信息
            logger.debug(f"Environment initialization: use_graph_mode={use_graph_mode}, device_target={device_target}")
            
            # 设置运行模式
            mode = context.GRAPH_MODE if use_graph_mode else context.PYNATIVE_MODE
            context.set_context(mode=mode)
            
            # 设置设备目标
            try:
                context.set_context(device_target=device_target)
                logger.debug("Successfully set MindSpore device target")
            except Exception as e:
                logger.warning(f"Failed to set device target to {device_target}, using CPU fallback: {e}")
                context.set_context(device_target='CPU')
        except ImportError:
            logger.warning("MindSpore not available, running in fallback mode")
        except Exception as e:
            logger.warning(f"Failed to configure MindSpore context: {e}")

        # 环境启用状态
        self.enabled = False

        # 自动混合精度类型
        autocast_dtype = None

        # 初始化随机数种子，使用PyTorch的初始种子
        _ms_seed = torch.initial_seed() % (1 << 31)
        # 使用线程本地存储保存运行时属性
        self._property = threading.local()
        self._property.content = [
            RuntimeProperty(
                mesh=_mesh, prng=_ms_seed, autocast_dtype=autocast_dtype)
        ]

    @property
    def param(self):
        return self._property.content[-1]

    def manual_seed(self, key):
        if isinstance(key, torch.Tensor):
            assert key.ndim == 0, 'manual seed can only take scalars'
            assert not key.dtype.is_floating_point, 'manual seed can only be integers'

            if isinstance(key, Tensor):
                key = key._elem
            else:
                key = key.item()

        # 设置MindSpore随机种子
        try:
            import mindspore as ms
            ms.set_seed(key)
        except ImportError:
            logger.warning("MindSpore not available, cannot set seed")

        # 更新运行时属性
        new_prop = self.param.override(prng=key)
        self._property.content.append(new_prop)

    @property
    def prng_key(self):
        return self.param.prng

    def _should_use_torch4ms_tensor(self, device):
        """
        判断是否应该使用torch4ms张量

        Args:
            device: 设备名称或设备对象

        Returns:
            bool: 如果应该使用torch4ms张量则返回True
        """
        if device is None:
            # 使用配置中的默认设备
            device = torch.get_default_device()

        # 标准化设备名称
        if isinstance(device, torch.device):
            device = device.type

        if ':' in device:
            device = device.split(':')[0]

        match device:
            case 'cpu':
                return False
            case 'cuda':
                return self.config.treat_cuda_as_mindspore_device
            case 'mindspore':
                return True
            case 'privateuseone':
                return True
            case 'meta':
                return self.enabled
        return False

    def load_ops(self):
        """
        加载所有注册的算子转换函数和分解函数

        从各个模块导入并注册操作到MindSpore的映射，包括：
        - 核心操作
        - 数学运算
        - 分解函数
        """
        # 导入算子实现模块
        # 注意：这里需要替换为MindSpore相关的操作实现
        from torch4ms.ops import maten, mtorch

        # 加载预注册的算子
        # 参考 torchax 的实现，根据 is_mindspore_function 标志分别存储
        for k, v in itertools.chain(ops_registry.all_aten_ops.items(),
                                    ops_registry.all_torch_functions.items()):
            if v.is_mindspore_function:
                # 存储MindSpore直接实现的操作（使用函数对象作为键）
                self._ops[k] = v
            else:
                # 存储需要分解的操作
                self._decomps[k] = v
        
        # 加载分解函数
        try:
            from torch4ms.decompositions import DECOMPOSITIONS, MUTABLE_DECOMPOSITION

            for k, v in DECOMPOSITIONS.items():
                op_name = _name_of_func(k) if not isinstance(k, str) else k
                if op_name not in self._decomps:
                    self._decomps[op_name] = ops_registry.Operator(
                        op_name,
                        v,
                        is_mindspore_function=False,
                        is_user_defined=False,
                        needs_env=False,
                        is_view_op=k in MUTABLE_DECOMPOSITION if not isinstance(k, str) else False,
                    )
        except ImportError:
            logger.warning("Failed to import decompositions module. Some operations may not be available.")

    def _get_op_or_decomp(self, func):
        """
        获取操作对应的MindSpore实现或分解函数
        
        参考 torchax 的实现，处理不同类型的操作符
        
        Args:
            func: PyTorch函数或操作符
            
        Returns:
            对应的Operator对象
            
        Raises:
            OperatorNotFound: 当找不到对应实现时
        """
        def _get_from_dict(op_dict, op):
            """从字典中查找操作，处理不同类型的操作符"""
            op = op_dict.get(func)
            # 处理OverloadPacket类型
            if op is None and isinstance(func, torch._ops.OpOverloadPacket):
                op = op_dict.get(func.default)
            # 处理OpOverload类型
            if op is None and isinstance(func, torch._ops.OpOverload):
                op = op_dict.get(func.overloadpacket)
            return op

        # 首先尝试从直接实现的操作中查找
        op = _get_from_dict(self._ops, func)

        if op is None:
            # 找不到直接实现时，尝试查找分解函数
            op = _get_from_dict(self._decomps, func)

        # 如果仍然找不到，抛出异常
        if op is None:
            raise OperatorNotFound(
                f"Operator with name {_name_of_func(func)} has no lowering")

        return op

    def _is_same_device(self, the_tensor, new_device):
        """
        检查张量是否与目标设备兼容

        Args:
            the_tensor: 要检查的张量
            new_device: 目标设备名称

        Returns:
            bool: 如果张量与设备兼容则返回True，否则返回False
        """
        # 如果没有指定设备，则视为兼容
        if new_device is None:
            return True

        # 标准化设备名称
        if ':' in str(new_device):
            new_device = str(new_device).split(':')[0]

        # 获取张量的设备信息
        tensor_device = str(the_tensor.device)
        if ':' in tensor_device:
            tensor_device = tensor_device.split(':')[0]

        # 检查设备类型是否匹配
        if tensor_device != new_device:
            # 特殊处理GPU设备
            if tensor_device == 'gpu' and new_device == 'cuda':
                return True  # MindSpore的GPU设备与PyTorch的CUDA兼容
            if tensor_device == 'cuda' and new_device == 'gpu':
                return True  # PyTorch的CUDA与MindSpore的GPU兼容
            return False
        return True

    def _to_copy(self, the_tensor, new_dtype, new_device):
        """
        处理张量的设备和数据类型转换

        Args:
            the_tensor: 要转换的张量
            new_dtype: 目标数据类型
            new_device: 目标设备

        Returns:
            转换后的张量
        """
        from mindspore import Tensor as MSTensor
        from mindspore import ops

        # 处理视图类型
        if isinstance(the_tensor, View):
            the_tensor = the_tensor.torch()

        # 标准化设备类型
        if new_device and not isinstance(new_device, str):
            new_device = str(new_device)

        res = the_tensor

        # 处理设备转换
        if not self._is_same_device(the_tensor, new_device):
            if isinstance(the_tensor, Tensor):
                # 从torch4ms张量转换到MindSpore张量并移动设备
                ms_tensor = the_tensor.mindspore()
                # 在MindSpore中移动设备
                res = ms_tensor.to(device=new_device)
                # 包装回torch4ms Tensor，安全获取requires_grad属性
                requires_grad = getattr(the_tensor, 'requires_grad', False)
                res = Tensor(res, self, requires_grad=requires_grad)
            elif isinstance(the_tensor, MSTensor):
                # 从MindSpore张量转换到torch4ms张量
                res = Tensor(the_tensor.to(device=new_device), self, requires_grad=False)
            else:
                # 对于其他类型，尝试直接转换
                try:
                    # 转换数据类型
                    if new_dtype is not None:
                        the_tensor = the_tensor.astype(new_dtype)
                    # 包装为torch4ms Tensor
                    res = Tensor(the_tensor, self, requires_grad=False)
                except Exception as e:
                    logger.warning(f"Failed to convert tensor to new device: {e}")
                    res = the_tensor

        # 处理数据类型转换
        if new_dtype is not None and hasattr(res, 'dtype') and res.dtype != new_dtype:
            if isinstance(res, Tensor):
                # 对torch4ms张量使用astype
                res = res.apply_mindspore(ops.cast, new_dtype)
            elif hasattr(res, 'astype'):
                # 对其他张量类型使用astype
                res = res.astype(new_dtype)

        return res

    def get_and_rotate_prng_key(self,
                                generator: Optional[torch.Generator] = None):
        """获取并旋转PRNG密钥

        Args:
            generator: 可选的PyTorch随机数生成器

        Returns:
            PRNG密钥
        """
        # 直接返回整数种子，在MindSpore中使用整数作为随机种子
        if generator is not None:
            return generator.initial_seed() % (2 ** 31)
        return self.param.get_and_rotate_prng_key()

    def _handle_tensor_constructor(self, op_name, args, kwargs, force_mindspore=False):
        """
        处理张量构造函数的调用

        Args:
            op_name: 操作名称字符串或函数对象
            args: 位置参数
            kwargs: 关键字参数
            force_mindspore: 强制使用torch4ms张量的标志

        Returns:
            创建的张量
        """
        # 规范化操作名称
        op_name_str = _name_of_func(op_name) if callable(op_name) else op_name
        
        # 获取设备参数
        device = kwargs.get("device")
        
        if force_mindspore:
            # 强制使用torch4ms张量
            should_use_torch4ms = True
        else:
            should_use_torch4ms = self._should_use_torch4ms_tensor(device)
        
        if should_use_torch4ms:
            # 移除device参数，避免PyTorch直接处理
            if device and str(device).lower() == 'mindspore':
                kwargs.pop('device', None)
            
            # 处理基本张量构造函数的特殊情况
            requires_grad = kwargs.get("requires_grad", False)
            res = None
            
            try:
                # 尝试获取操作对应的实现
                op = self._get_op_or_decomp(op_name_str)
                
                # 设置环境参数
                if op.needs_env:
                    kwargs['env'] = self
                
                # 转换参数到MindSpore格式
                if op.is_mindspore_function:
                    (args, kwargs) = self.t2ms_iso((args, kwargs))
                
                # 执行操作
                res = op.func(*args, **kwargs)
                
                # 转换结果为torch4ms Tensor
                res = self.ms2t_iso(res)
            except OperatorNotFound:
                # 为基本张量构造函数添加直接实现
                if op_name_str == 'tensor':
                    # 处理torch.tensor构造函数
                    data = args[0] if args else kwargs.get('data')
                    dtype = kwargs.get('dtype')
                    
                    # 将数据转换为MindSpore张量
                    if isinstance(data, (list, tuple)):
                        data = np.array(data)
                    if isinstance(data, np.ndarray):
                        # 如果提供了dtype，转换为对应类型
                        if dtype:
                            data = data.astype(mappings.t2ms_dtype(dtype))
                        res = ms_Tensor(data)
                    elif isinstance(data, (int, float)):
                        res = ms_Tensor(np.array([data]))
                    elif isinstance(data, torch.Tensor):
                        # 从PyTorch张量转换
                        res = ms_Tensor(data.detach().numpy())
                elif op_name_str == 'ones':
                    # 处理torch.ones构造函数
                    size = args[0] if args else kwargs.get('size')
                    dtype = kwargs.get('dtype', torch.float32)
                    # 先创建默认类型的numpy数组，然后转换为MindSpore张量
                    np_array = np.ones(size, dtype=np.float32)
                    res = ms_Tensor(np_array)
                    # 如果提供了dtype，转换为对应类型
                    if dtype in mappings.TORCH_DTYPE_TO_MINDSPORE:
                        res = res.astype(mappings.TORCH_DTYPE_TO_MINDSPORE[dtype])
                elif op_name_str == 'zeros':
                    # 处理torch.zeros构造函数
                    size = args[0] if args else kwargs.get('size')
                    dtype = kwargs.get('dtype', torch.float32)
                    # 先创建默认类型的numpy数组，然后转换为MindSpore张量
                    np_array = np.zeros(size, dtype=np.float32)
                    res = ms_Tensor(np_array)
                    # 如果提供了dtype，转换为对应类型
                    if dtype in mappings.TORCH_DTYPE_TO_MINDSPORE:
                        res = res.astype(mappings.TORCH_DTYPE_TO_MINDSPORE[dtype])
                elif op_name_str == 'empty':
                    # 处理torch.empty构造函数
                    size = args[0] if args else kwargs.get('size')
                    dtype = kwargs.get('dtype', torch.float32)
                    # 先创建默认类型的numpy数组，然后转换为MindSpore张量
                    np_array = np.empty(size, dtype=np.float32)
                    res = ms_Tensor(np_array)
                    # 如果提供了dtype，转换为对应类型
                    if dtype in mappings.TORCH_DTYPE_TO_MINDSPORE:
                        res = res.astype(mappings.TORCH_DTYPE_TO_MINDSPORE[dtype])
                elif op_name_str == 'rand':
                    # 处理torch.rand构造函数
                    size = args[0] if args else kwargs.get('size')
                    dtype = kwargs.get('dtype', torch.float32)
                    # 先创建默认类型的numpy数组，然后转换为MindSpore张量
                    np_array = np.random.rand(*size).astype(np.float32)
                    res = ms_Tensor(np_array)
                    # 如果提供了dtype，转换为对应类型
                    if dtype in mappings.TORCH_DTYPE_TO_MINDSPORE:
                        res = res.astype(mappings.TORCH_DTYPE_TO_MINDSPORE[dtype])
                elif op_name_str == 'randn':
                    # 处理torch.randn构造函数
                    # torch.randn 可以接受多个位置参数或一个size元组
                    if args and isinstance(args[0], (tuple, list)):
                        size = args[0]
                    elif args:
                        # 多个位置参数，如 torch.randn(2, 3, 10)
                        size = args
                    else:
                        size = kwargs.get('size', ())
                    dtype = kwargs.get('dtype', torch.float32)
                    # 先创建默认类型的numpy数组，然后转换为MindSpore张量
                    np_array = np.random.randn(*size).astype(np.float32)
                    res = ms_Tensor(np_array)
                    # 如果提供了dtype，转换为对应类型
                    if dtype in mappings.TORCH_DTYPE_TO_MINDSPORE:
                        res = res.astype(mappings.TORCH_DTYPE_TO_MINDSPORE[dtype])
                elif op_name_str == 'full':
                    # 处理torch.full构造函数
                    size = args[0] if len(args) > 1 else kwargs.get('size')
                    fill_value = args[1] if len(args) > 1 else kwargs.get('fill_value')
                    dtype = kwargs.get('dtype')
                    if dtype:
                        fill_value = mappings.t2ms_dtype(dtype)(fill_value)
                    res = ms_Tensor(np.full(size, fill_value))
                elif op_name_str == 'arange':
                    # 处理torch.arange构造函数
                    start = args[0] if len(args) > 0 else kwargs.get('start', 0)
                    end = args[1] if len(args) > 1 else kwargs.get('end')
                    step = args[2] if len(args) > 2 else kwargs.get('step', 1)
                    dtype = kwargs.get('dtype')
                    if dtype:
                        dtype = mappings.t2ms_dtype(dtype)
                    else:
                        dtype = np.float32
                    res = ms_Tensor(np.arange(start, end, step, dtype=dtype))
            
            # 包装结果为torch4ms张量
            if isinstance(res, ms_Tensor):
                res = Tensor(res, self, requires_grad)
            
            return res
        else:
            # 如果不应该使用torch4ms张量，使用原生PyTorch处理
            with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                # 获取原始PyTorch函数
                if isinstance(op_name, str):
                    # 从torch模块获取对应函数
                    if hasattr(torch, op_name_str):
                        func = getattr(torch, op_name_str)
                    else:
                        raise RuntimeError(f"PyTorch function {op_name_str} not found")
                else:
                    func = op_name
                
                return func(*args, **kwargs)

    def _tensor_to(self, args, kwargs):
        """
        处理张量的to方法

        Args:
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            转换后的张量
        """
        # 获取目标张量
        the_tensor = args[0] if args else None
        if the_tensor is None:
            raise ValueError("Tensor 'to' method requires a tensor as first argument")

        # 处理参数，过滤掉布尔值参数
        filtered_args = list(filter(lambda x: not isinstance(x, bool), args[1:]))

        # 解析device和dtype参数
        device = None
        dtype = None

        if len(filtered_args) >= 1:
            if isinstance(filtered_args[0], str) or hasattr(filtered_args[0], 'device_id'):
                device = filtered_args[0]
            elif hasattr(filtered_args[0], 'type'):
                dtype = filtered_args[0]

        if len(filtered_args) >= 2:
            if isinstance(filtered_args[1], str) or hasattr(filtered_args[1], 'device_id'):
                device = filtered_args[1]
            elif hasattr(filtered_args[1], 'type'):
                dtype = filtered_args[1]

        # 从kwargs中获取device和dtype
        if 'device' in kwargs:
            device = kwargs['device']
        if 'dtype' in kwargs:
            dtype = kwargs['dtype']

        return self._to_copy(the_tensor, dtype, device)

    def dispatch(self, op_name, types=None, args=(), kwargs=None):
        """
        核心分发函数，处理操作的转换和执行

        Args:
            op_name: 操作名称或函数对象
            types: 参数类型列表
            args: 位置参数
            kwargs: 关键字参数

        Returns:
            操作执行结果
        """
        kwargs = kwargs or {}

        # 规范化操作名称
        op_name_str = _name_of_func(op_name)

        # 特殊处理张量构造函数
        if op_name_str in TENSOR_CONSTRUCTORS:
            return self._handle_tensor_constructor(op_name_str, args, kwargs)

        # 特殊处理to方法和相关操作
        # 匹配完整名称或不带命名空间的名称
        base_op_name = op_name_str.split(".")[-1] if "." in op_name_str else op_name_str
        if op_name_str in ("to", "_to_copy", "lift_fresh") or base_op_name in ("to", "_to_copy", "lift_fresh"):
            return self._tensor_to(args, kwargs)

        # 如果函数不作用于张量且不是构造函数，尝试使用兼容层
        tensor_args = [
            t for t in list(args) if isinstance(t, (Tensor, ms_Tensor))
        ]

        def is_not_torch4ms_tensor(x):
            return not isinstance(x, Tensor) and not isinstance(x, View)

        # 如果所有张量参数都不是torch4ms张量，尝试使用原生处理
        if tensor_args and all(is_not_torch4ms_tensor(t) for t in tensor_args):
            # 这里可以添加兼容层处理或抛出异常
            raise NotImplementedError(f"Native execution not implemented for {op_name_str}")

        # 使用MindSpore的方式处理操作
        try:
            # 获取操作对应的实现（传递函数对象而不是字符串）
            op = self._get_op_or_decomp(op_name)

            # 保存原始参数用于调试
            old_args, old_kwargs = args, kwargs

            try:
                if not op.is_view_op:
                    args, kwargs = self.v2t_iso((args, kwargs))

                with self:
                    if self.param.autocast_dtype is not None:
                        # MindSpore的自动混合精度处理
                        autocast_policy = amp.autocast_policy.get(op_name_str)
                        if autocast_policy is not None:
                            args, kwargs = amp.execute_policy(autocast_policy, args, kwargs,
                                                              self.param.autocast_dtype)

                if op.is_mindspore_function:
                    args, kwargs = self.t2ms_iso((args, kwargs))
            except AssertionError:
                if self.config.debug_mixed_tensor:
                    # 在调试模式下进入断点
                    print(f"Assertion error in dispatch for {op_name_str}")
                    import pdb  # 引入调试模块
                    pdb.set_trace()  # 进入调试断点
                else:
                    raise
            except Exception as e:  # 新增通用异常处理
                if self.config.debug_mixed_tensor:
                    print(f"Unexpected error in dispatch for {op_name_str}: {str(e)}")
                    import pdb
                    pdb.set_trace()
                else:
                    raise

            if op.needs_env:
                kwargs["env"] = self

            if op.is_mindspore_function:
                res = op.func(*args, **kwargs)
            else:
                # 执行非MindSpore函数
                res = op.func(*args, **kwargs)

            if op.is_mindspore_function:
                res = self.ms2t_iso(res)

            if self.config.force_materialize_views and isinstance(res, View):
                res = res.torch()

            if self.config.debug_accuracy_for_each_op:
                debug_accuracy(op_name_str, old_args, old_kwargs, res)
            return res
        except OperatorNotFound:
            # 处理算子未找到的情况
            raise
        except Exception as e:
            # 处理其他可能的异常
            logger.error(f"Error during dispatch of {op_name_str}: {str(e)}")
            raise

    def enable_mindspore_handlers(self):
        """
        启用MindSpore操作处理机制

        参考 torchax 的实现，启用 PyTorch 函数模式和分发模式来拦截操作。
        """
        # 进入分发模式和函数模式（参考 torchax 的实现）
        self._dispatch_mode.__enter__()
        self._function_mode.__enter__()
        # 标记环境为启用状态
        self.enabled = True

    def disable_mindspore_handlers(self, *exc):
        """
        禁用MindSpore操作处理机制

        参考 torchax 的实现，禁用 PyTorch 函数模式和分发模式。

        Args:
            *exc: 异常信息（类型、值、回溯）
        """
        # 如果没有提供异常信息，设置默认值
        if not exc:
            exc = (None, None, None)
        # 退出函数模式和分发模式（参考 torchax 的实现）
        self._function_mode.__exit__(*exc)
        self._dispatch_mode.__exit__(*exc)
        # 标记环境为禁用状态
        self.enabled = False

    # 为了兼容性，保留旧的方法名称
    enable_torch_modes = enable_mindspore_handlers
    disable_torch_modes = disable_mindspore_handlers

    def __enter__(self):
        self.enable_torch_modes()
        # 推入PyTorch分发模式
        self._dispatch_mode.__enter__()
        return self

    def __exit__(self, *exc):
        """
        上下文管理器的退出方法

        禁用PyTorch模式，清理资源。

        Args:
            *exc: 异常信息（类型、值、回溯）
        """
        # 退出PyTorch分发模式
        self._dispatch_mode.__exit__(*exc)
        self.disable_torch_modes(*exc)

    def _move_one_value(self, val):
        """
        将单个值移动到MindSpore环境

        Args:
            val: 要移动的值（可以是模块、张量或其他类型）

        Returns:
            转换为torch4ms环境的值
        """
        # 处理神经网络模块
        # MindSpore的Module处理逻辑需要单独实现
        # 这里只是简单的占位符
        # if hasattr(val, 'to'):
        #   with self:
        #     return val.to("mindspore")

        # 已经是torch4ms张量，直接返回
        if isinstance(val, Tensor):
            return val

        # MindSpore张量转换为torch4ms张量
        from mindspore import Tensor as MSTensor
        if isinstance(val, MSTensor):
            return Tensor(val, self)

        # 非张量值不进行转换
        return val

    def to_xla(self, values):
        """
        将值树结构转换为torch4ms环境中的值

        注意：这里支持MindSpore张量和其他类型的转换

        Args:
            values: 包含各种值的树结构

        Returns:
            转换后的树结构，其中MindSpore张量转换为torch4ms.Tensor
        """

        # 简单的递归映射实现，替代torch_pytree
        def tree_map(obj, map_fn):
            if isinstance(obj, (list, tuple)):
                return type(obj)(tree_map(item, map_fn) for item in obj)
            elif isinstance(obj, dict):
                return {k: tree_map(v, map_fn) for k, v in obj.items()}
            return map_fn(obj)

        # 使用tree_map递归地转换树结构中的每个值
        res = tree_map(values, self._move_one_value)
        return res

    def t2ms_iso(self, tensors):
        """将torch4ms Tensor转换为MindSpore张量

        此函数不会复制数据，只是简单地解包内部的MindSpore张量
        注意：iso是"isomorphic"（同构）的缩写

        Args:
            tensors: 要转换的张量或包含张量的树结构

        Returns:
            转换后的MindSpore张量或包含MindSpore张量的树结构
        """

        def to_mindspore(x):
            # 如果是torch4ms张量或视图，获取内部MindSpore张量（优先处理，避免递归）
            if isinstance(x, Tensor) or isinstance(x, View):
                return x.mindspore()
            # 如果是普通torch.Tensor，处理标量张量的特殊情况
            if isinstance(x, torch.Tensor):
                # 处理标量张量的特殊情况（避免调用可能触发dispatch的方法）
                if self.config.allow_mixed_math_with_scalar_tensor:
                    # 直接检查ndim，避免调用squeeze()导致递归
                    if x.ndim == 0 or (x.ndim > 0 and all(s == 1 for s in x.shape)):
                        # 使用no_dispatch避免触发dispatch
                        with mode_utils.no_dispatch(), torch._C.DisableTorchFunction():
                            if x.ndim == 0:
                                return x.item()
                            # 对于所有维度都是1的张量，也提取标量值
                            squeezed = x.squeeze()
                            if squeezed.ndim == 0:
                                return squeezed.item()
                return self.t2ms_copy(x)
            # 其他情况抛出异常
            raise TypeError(
                f"Expect a Tensor, View, or torch.Tensor but got {type(x)}; usually this means there is a mixed math between MindSporeTensor and other tensor types"
            )

        # 使用 torch_pytree 进行递归映射（参考 torchax 的实现）
        import torch.utils._pytree as torch_pytree
        return torch_pytree.tree_map_only(torch.Tensor, to_mindspore, tensors)

    def v2t_iso(self, views):

        def to_tensor(x):
            if isinstance(x, View):
                return x.torch()
            return x

        res = torch_pytree.tree_map_only(View, to_tensor, views)
        return res

    def ms2t_iso(self, ms_values):
        """将MindSpore张量转换为torch4ms Tensor

        此函数不会复制数据，只是用torch4ms Tensor包装MindSpore张量
        注意：iso是"isomorphic"（同构）的缩写

        Args:
            ms_values: 要转换的MindSpore张量或包含MindSpore张量的树结构

        Returns:
            转换后的torch4ms.Tensor或包含torch4ms.Tensor的树结构
        """

        # 使用 torch_pytree 进行递归映射（参考 torchax 的实现）
        import torch.utils._pytree as torch_pytree
        from mindspore import Tensor as MSTensor
        
        def to_torch4ms(x):
            if isinstance(x, MSTensor):
                return Tensor(x, self)
            return x

        return torch_pytree.tree_map_only(MSTensor, to_torch4ms, ms_values)

    def ms2t_copy(self, args):
        """将MindSpore张量转换为CPU上的PyTorch张量

        此操作可能涉及数据复制（取决于是否启用DLPack）

        Args:
            args: 包含MindSpore张量的树结构

        Returns:
            包含PyTorch张量的树结构
        """
        from mindspore import Tensor as MSTensor
        return torch_pytree.tree_map_only(
            MSTensor,
            lambda x: mappings.ms2t(x, self.config.use_dlpack_for_data_conversion),
            args)

    def t2ms_copy(self, args):
        """将CPU上的PyTorch张量转换为MindSpore张量

        此操作可能涉及数据复制（取决于是否启用DLPack）

        Args:
            args: 包含PyTorch张量的树结构

        Returns:
            包含MindSpore张量的树结构
        """
        return torch_pytree.tree_map_only(
            torch.Tensor,
            lambda x: mappings.t2ms(x, self.config.use_dlpack_for_data_conversion),
            args)

    def override_op_definition(self, op_to_override, op_impl):
        """
        覆盖操作的定义

        允许用户替换已注册的操作实现，用于自定义算子转换逻辑。

        Args:
            op_to_override: 要覆盖的操作
            op_impl: 新的操作实现函数
        """
        # 创建新的操作对象并注册
        self._ops[op_to_override] = ops_registry.Operator(
            op_to_override,
            op_impl,
            is_mindspore_function=False,
            is_user_defined=True,
            needs_env=False,
            is_view_op=False
        )

    @contextlib.contextmanager
    def override_property(self, **kwargs):
        """
        临时覆盖运行时属性的上下文管理器

        允许用户在特定上下文中临时修改环境的运行时属性，如mesh、prng等。
        上下文结束后，属性将恢复到之前的值。

        Args:
            **kwargs: 要覆盖的属性及其新值
        """
        # 创建新的属性对象并将新属性添加到属性栈中
        new_prop = self.param.override(**kwargs)
        self._property.content.append(new_prop)
        yield
        self._property.content.pop()