import sys
import os
import pytest

# Ensure src/ is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Set device mode for testing (must be done before importing mindtorch_v2)
# Use MINDTORCH_TEST_DEVICE env var to override (default: CPU for CI compatibility)
import mindspore
device = os.environ.get('MINDTORCH_TEST_DEVICE', 'CPU')
mindspore.set_context(device_target=device)


@pytest.fixture(autouse=True)
def reset_device_context():
    """Reset device context state after each test to prevent pollution."""
    yield
    # Reset the thread-local device context after each test
    try:
        from mindtorch_v2._device import _device_context
        if hasattr(_device_context, 'device'):
            delattr(_device_context, 'device')
    except (ImportError, AttributeError):
        pass
