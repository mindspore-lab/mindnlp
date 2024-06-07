#################################################################################################
# Copyright (c) 2022-2024 Ali Hassani.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
#################################################################################################
import enum
import logging
import os

log_format = "| %(asctime)s | [[ %(name)s ]] [ %(levelname)s ]: %(message)s"


class LogLevel(enum.Enum):
    Default = 0
    Debug = 1
    Info = 2
    Warnings = 3
    Errors = 4
    Critical = 5


def _get_log_level() -> LogLevel:
    if "NATTEN_LOG_LEVEL" in os.environ:
        log_level = str(os.environ["NATTEN_LOG_LEVEL"]).lower()
        if log_level == "debug":
            return LogLevel.Debug
        elif log_level == "info":
            return LogLevel.Info
        elif log_level == "warning":
            return LogLevel.Warnings
        elif log_level == "error":
            return LogLevel.Errors
        elif log_level == "critical":
            return LogLevel.Critical

    return LogLevel.Default


_map_log_level = {
    LogLevel.Default: logging.INFO,
    LogLevel.Debug: logging.DEBUG,
    LogLevel.Info: logging.INFO,
    LogLevel.Warnings: logging.WARNING,
    LogLevel.Errors: logging.ERROR,
    LogLevel.Critical: logging.CRITICAL,
}


def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(_map_log_level[_get_log_level()])
    formatter = logging.Formatter(log_format)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
