"""
MindSpore configuration utilities.
This module provides functions to configure and manage MindSpore environment.
"""
import os
import mindspore as ms
import mindspore.context as context
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default MindSpore configuration
DEFAULT_MS_CONFIG = {
    'device_target': 'CPU',  # Default to CPU, can be 'GPU' or 'Ascend'
    'mode': context.GRAPH_MODE,  # Default to graph mode, can be PYNATIVE_MODE
    'device_id': 0,  # Default device ID
    'enable_graph_kernel': False,  # Whether to enable graph kernel optimization
}

# Current configuration
_current_config = DEFAULT_MS_CONFIG.copy()

def set_ms_config(**kwargs):
    """
    Set MindSpore configuration.
    
    Args:
        **kwargs: Configuration parameters to set.
            device_target: Target device type ('CPU', 'GPU', 'Ascend').
            mode: Running mode (GRAPH_MODE or PYNATIVE_MODE).
            device_id: Device ID to use.
            enable_graph_kernel: Whether to enable graph kernel optimization.
    """
    global _current_config
    
    # Update configuration
    _current_config.update(kwargs)
    
    # Apply configuration
    try:
        # Set device target
        if 'device_target' in kwargs:
            context.set_context(device_target=kwargs['device_target'])
            logger.info(f"Set device target to {kwargs['device_target']}")
        
        # Set running mode
        if 'mode' in kwargs:
            context.set_context(mode=kwargs['mode'])
            logger.info(f"Set running mode to {kwargs['mode']}")
        
        # Set device ID
        if 'device_id' in kwargs:
            context.set_context(device_id=kwargs['device_id'])
            logger.info(f"Set device ID to {kwargs['device_id']}")
        
        # Set graph kernel optimization
        if 'enable_graph_kernel' in kwargs:
            context.set_context(enable_graph_kernel=kwargs['enable_graph_kernel'])
            logger.info(f"Set graph kernel optimization to {kwargs['enable_graph_kernel']}")
    except Exception as e:
        logger.error(f"Failed to set MindSpore configuration: {e}")
        raise

def get_ms_config():
    """
    Get current MindSpore configuration.
    
    Returns:
        dict: Current configuration.
    """
    return _current_config.copy()

def initialize_ms():
    """
    Initialize MindSpore with current configuration.
    """
    try:
        # Apply all current configuration settings
        context.set_context(
            device_target=_current_config['device_target'],
            mode=_current_config['mode'],
            device_id=_current_config['device_id'],
            enable_graph_kernel=_current_config['enable_graph_kernel']
        )
        logger.info(f"Initialized MindSpore with configuration: {_current_config}")
    except Exception as e:
        logger.error(f"Failed to initialize MindSpore: {e}")
        raise

def is_ms_available():
    """
    Check if MindSpore is available.
    
    Returns:
        bool: True if MindSpore is available, False otherwise.
    """
    try:
        import mindspore
        return True
    except ImportError:
        return False

def get_ms_device_type():
    """
    Get the current MindSpore device type.
    
    Returns:
        str: Device type ('CPU', 'GPU', 'Ascend').
    """
    return _current_config['device_target']

def set_precision_mode(precision_mode='fp32'):
    """
    Set precision mode for MindSpore.
    
    Args:
        precision_mode: Precision mode ('fp32', 'fp16', 'mixed').
    """
    if precision_mode not in ['fp32', 'fp16', 'mixed']:
        raise ValueError(f"Invalid precision mode: {precision_mode}. Must be one of ['fp32', 'fp16', 'mixed']")
    
    try:
        context.set_context(precision_mode=precision_mode)
        logger.info(f"Set precision mode to {precision_mode}")
    except Exception as e:
        logger.error(f"Failed to set precision mode: {e}")
        raise

def enable_parallel(mode=True):
    """
    Enable or disable parallel mode.
    
    Args:
        mode: Whether to enable parallel mode.
    """
    try:
        if mode:
            context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.AUTO_PARALLEL)
            logger.info("Enabled auto parallel mode")
        else:
            context.set_auto_parallel_context(parallel_mode=ms.ParallelMode.STAND_ALONE)
            logger.info("Disabled parallel mode")
    except Exception as e:
        logger.error(f"Failed to set parallel mode: {e}")
        raise