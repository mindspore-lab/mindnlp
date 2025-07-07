from mindspore.profiler import ProfilerLevel as msProfilerLevel
from mindspore.profiler import AicoreMetrics


class ProfilerLevel:
    Level0 = msProfilerLevel.Level0
    Level1 = msProfilerLevel.Level1
    Level2 = msProfilerLevel.Level2
    Level_none = msProfilerLevel.LevelNone


class AiCMetrics:
    PipeUtilization = AicoreMetrics.PipeUtilization
    ArithmeticUtilization = AicoreMetrics.ArithmeticUtilization
    Memory = AicoreMetrics.Memory
    MemoryL0 = AicoreMetrics.MemoryL0
    MemoryUB = AicoreMetrics.MemoryUB
    ResourceConflictRatio = AicoreMetrics.ResourceConflictRatio
    L2Cache = AicoreMetrics.L2Cache
    MemoryAccess = AicoreMetrics.AiCoreNone # ToImplenmentened
    AiCoreNone = AicoreMetrics.AiCoreNone


class ExportType:
    Db = "db"
    Text = "text"


class _ExperimentalConfig:
    def __init__(self,
                 profiler_level: int = ProfilerLevel.Level0,
                 aic_metrics: int = AiCMetrics.AiCoreNone,
                 l2_cache: bool = False,
                 msprof_tx: bool = False,
                 data_simplification: bool = True,
                 record_op_args: bool = False,
                 op_attr: bool = False,
                 gc_detect_threshold: float = None,
                 export_type: str = ExportType.Text):
        self._profiler_level = profiler_level
        self._aic_metrics = aic_metrics
        if self._profiler_level != None:
            if self._profiler_level != ProfilerLevel.Level0 and self._aic_metrics == AiCMetrics.AiCoreNone:
                self._aic_metrics = AiCMetrics.PipeUtilization
        self._l2_cache = l2_cache
        self._msprof_tx = msprof_tx
        self._data_simplification = data_simplification
        self.record_op_args = record_op_args
        self._export_type = export_type
        self._op_attr = op_attr
        self._gc_detect_threshold = gc_detect_threshold
