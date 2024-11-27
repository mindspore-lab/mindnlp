"""""devices manager for mindnlp"""
import acl
from loguru import logger

def check_Ascend_npu_available():
    """check available Ascend npus

    Returns:
        is_npu_avaliable: True
        device_count: count of Ascend available npus
    """
    ret = acl.init()
    if ret!= 0:
        logger.error("ACL initialization fails, NPU may be unavailable, please check the relevant environment and device connection!")
        return False, 0
    else:
        device_count = acl.get_device_count()
        if device_count == 0:
            logger.error("If the ACL is initialized, the Ascend NPU may be available.")
            acl.finalize()
            return False, acl.get_device_count()
        else:
            logger.info(f"{device_count} Ascend NPU devices are detected, and the NPU is available.")
            acl.finalize()
            return True, device_count

# _is_Ascend_npu_avaliable, _avaliable_Ascend_npus_count = check_Ascend_npu_available() # acl is a problem