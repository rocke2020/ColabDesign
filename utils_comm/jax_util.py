import os
import sys

import jax

sys.path.append(os.path.abspath("."))
from utils_comm.log_util import logger


def check_gpu_count():
    """Returns: 0 means no gpu"""
    if jax.default_backend() == "gpu":
        gpu_count = jax.local_device_count()
        logger.info("gpu_count %s", gpu_count)
        return gpu_count
    else:
        logger.exception("Warning: no gpu, use CPU!!")
        return 0


def get_gpu_device_id(gpu_device_id: int):
    """if gpu_device_id >= gpu_count, that's too large, use gpu_device_id = '0'"""
    gpu_count = check_gpu_count()
    if gpu_device_id >= gpu_count:
        gpu_device_id = 0
    logger.info("gpu_device_id %s", gpu_device_id)
    return gpu_device_id


if __name__ == "__main__":
    check_gpu_count()
