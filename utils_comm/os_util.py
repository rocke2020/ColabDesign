import logging
import time

import psutil

logger = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, datefmt='%y-%m-%d %H:%M',
    format='%(asctime)s %(filename)s %(lineno)d: %(message)s')


def kill_process_with_all_sub(pid):
    """  """
    check_process_status(pid)
    logger.info(f'killing pid {pid}')
    for child in psutil.Process(pid).children(recursive=True):
        logger.info(f'killing child {child.name}')
        child.kill()
        time.sleep(0.1)
    psutil.Process(pid).kill()
    logger.info('check status after killed all sub processes')
    check_process_status(pid)


def kill_multi_prcoesss(ids: list[int]):
    """  """
    for pid in ids:
        try:
            kill_process_with_all_sub(pid)
        except Exception as identifier:
            logger.info('%s', identifier)


def check_process_status(pid):
    """  """
    if not psutil.pid_exists(pid):
        logger.info(f'pid {pid} does not exist')
        return
    logger.info(psutil.Process(pid))


## limit memory usage of current process
# p = psutil.Process()
# print(p.pid)
# def limit_memory(maxsize):
#     soft,hard = resource.getrlimit(resource.RLIMIT_AS)
#     resource.setrlimit(resource.RLIMIT_AS,(maxsize,hard))

# limit_memory(1024*1024*120*1024)


if __name__ == "__main__":
    ids = [
        679116,
        # 672068,
    ]
    kill_multi_prcoesss(ids)
    # check_process_status(786967)