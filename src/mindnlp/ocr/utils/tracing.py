# -*- coding: utf-8 -*-
"""
分布式追踪模- OpenTelemetry集成

实现全链路追
- FastAPI 自动埋点
- 自定Span 标记
- 集成 Jaeger/Zipkin
- 采样率配
"""

import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

from opentelemetry import trace  # pylint: disable=import-error
from opentelemetry.sdk.trace import TracerProvider  # pylint: disable=import-error
from opentelemetry.sdk.trace.export import BatchSpanProcessor  # pylint: disable=import-error
from opentelemetry.sdk.resources import Resource, SERVICE_NAME  # pylint: disable=import-error
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter  # pylint: disable=import-error
from opentelemetry.exporter.jaeger.thrift import JaegerExporter  # pylint: disable=import-error
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # pylint: disable=import-error
from opentelemetry.sdk.trace.sampling import TraceIdRatioBased  # pylint: disable=import-error

# 使用标准logger避免循环导入
logger = logging.getLogger(__name__)


class TracingConfig:
    """追踪配置"""

    def __init__(
        self,
        enabled: bool = True,
        service_name: str = "ocr-api",
        sampling_rate: float = 0.1,
        exporter_type: str = "jaeger",  # jaeger, otlp, console
        jaeger_endpoint: Optional[str] = None,
        otlp_endpoint: Optional[str] = None,
    ):
        """
        Args:
            enabled: 是否启用追踪
            service_name: 服务名称
            sampling_rate: 采样(0.0-1.0)
            exporter_type: 导出器类
            jaeger_endpoint: Jaeger端点
            otlp_endpoint: OTLP端点
        """
        self.enabled = enabled
        self.service_name = service_name
        self.sampling_rate = sampling_rate
        self.exporter_type = exporter_type
        self.jaeger_endpoint = jaeger_endpoint or "localhost:6831"
        self.otlp_endpoint = otlp_endpoint or "http://localhost:4317"


def setup_tracing(config: TracingConfig) -> Optional[TracerProvider]:
    """
    配置分布式追

    Args:
        config: 追踪配置

    Returns:
        TracerProvider实例
    """
    if not config.enabled:
        logger.info("Distributed tracing is disabled")
        return None

    # 创建资源
    resource = Resource(attributes={
        SERVICE_NAME: config.service_name
    })

    # 创建采样
    sampler = TraceIdRatioBased(config.sampling_rate)

    # 创建 TracerProvider
    provider = TracerProvider(
        resource=resource,
        sampler=sampler
    )

    # 创建导出
    if config.exporter_type == "jaeger":
        exporter = JaegerExporter(
            agent_host_name=config.jaeger_endpoint.split(":")[0],
            agent_port=int(config.jaeger_endpoint.split(":")[1]),
        )
        logger.info(f"Using Jaeger exporter: {config.jaeger_endpoint}")

    elif config.exporter_type == "otlp":
        exporter = OTLPSpanExporter(
            endpoint=config.otlp_endpoint,
            insecure=True  # 使用 HTTP
        )
        logger.info(f"Using OTLP exporter: {config.otlp_endpoint}")

    elif config.exporter_type == "console":
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # pylint: disable=import-error
        exporter = ConsoleSpanExporter()
        logger.info("Using Console exporter (for debugging)")

    else:
        logger.warning(f"Unknown exporter type: {config.exporter_type}, using console")
        from opentelemetry.sdk.trace.export import ConsoleSpanExporter  # pylint: disable=import-error
        exporter = ConsoleSpanExporter()

    # 添加批处理器
    provider.add_span_processor(BatchSpanProcessor(exporter))

    # 设置全局 tracer
    trace.set_tracer_provider(provider)

    logger.info(
        f"Distributed tracing configured: "
        f"service={config.service_name}, "
        f"sampling_rate={config.sampling_rate}, "
        f"exporter={config.exporter_type}"
    )

    return provider


def instrument_fastapi(app):
    """
    FastAPI 应用添加自动埋点

    Args:
        app: FastAPI 应用实例
    """
    try:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("FastAPI instrumentation enabled")
    except Exception as e:
        logger.error(f"Failed to instrument FastAPI: {e}")


def get_tracer(name: str = "ocr") -> trace.Tracer:
    """
    获取追踪

    Args:
        name: 追踪器名

    Returns:
        Tracer实例
    """
    return trace.get_tracer(name)


@contextmanager
def trace_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
    kind: trace.SpanKind = trace.SpanKind.INTERNAL
):
    """
    创建追踪 Span 的上下文管理

    Args:
        name: Span名称
        attributes: Span属
        kind: Span类型

    Example:
        with trace_span("image_preprocessing", {"image_size": "1024x768"}):
            # 处理图像
            pass
    """
    tracer = get_tracer()

    with tracer.start_as_current_span(name, kind=kind) as span:
        # 添加属
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, value)

        try:
            yield span
        except Exception as e:
            # 记录异常
            span.record_exception(e)
            span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
            raise


class OCRTracer:
    """
    OCR 专用追踪

    提供便捷的追踪方
    """

    def __init__(self, tracer_name: str = "ocr"):
        """
        Args:
            tracer_name: 追踪器名
        """
        self.tracer = get_tracer(tracer_name)

    @contextmanager
    def trace_request(self, request_id: str, endpoint: str):
        """
        追踪HTTP请求

        Args:
            request_id: 请求ID
            endpoint: 请求端点
        """
        with self.tracer.start_as_current_span(
            "http_request",
            kind=trace.SpanKind.SERVER
        ) as span:
            span.set_attribute("request.id", request_id)
            span.set_attribute("http.route", endpoint)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_preprocessing(self, image_size: str, format: str):
        """
        追踪图像预处

        Args:
            image_size: 图像大小
            format: 图像格式
        """
        with self.tracer.start_as_current_span("preprocessing") as span:
            span.set_attribute("image.size", image_size)
            span.set_attribute("image.format", format)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_inference(
        self,
        model_name: str,
        batch_size: int,
        max_tokens: int
    ):
        """
        追踪模型推理

        Args:
            model_name: 模型名称
            batch_size: 批次大小
            max_tokens: 最大token
        """
        with self.tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("model.name", model_name)
            span.set_attribute("model.batch_size", batch_size)
            span.set_attribute("model.max_tokens", max_tokens)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    @contextmanager
    def trace_postprocessing(self, output_format: str):
        """
        追踪后处

        Args:
            output_format: 输出格式
        """
        with self.tracer.start_as_current_span("postprocessing") as span:
            span.set_attribute("output.format", output_format)

            try:
                yield span
            except Exception as e:
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """
        在当Span 添加事件

        Args:
            name: 事件名称
            attributes: 事件属
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.add_event(name, attributes or {})

    def set_attribute(self, key: str, value: Any):
        """
        在当Span 设置属

        Args:
            key: 属性键
            value: 属性
        """
        current_span = trace.get_current_span()
        if current_span.is_recording():
            current_span.set_attribute(key, value)


# 全局追踪器实
_ocr_tracer: Optional[OCRTracer] = None


def get_ocr_tracer() -> OCRTracer:
    """获取OCR追踪器"""
    global _ocr_tracer
    if _ocr_tracer is None:
        _ocr_tracer = OCRTracer()
    return _ocr_tracer
