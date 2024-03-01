""" Basic usage example: 

from utils_comm.log_util import logger

logger.info('comments')
arr = [[1, 2], [3, 4]]
logger.info(f'len(arr) {len(arr)}')
logger.info('len(arr) %s', len(arr))
"""

import contextlib
import logging
import os
import time
from datetime import datetime
from pathlib import Path

from icecream import ic

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


FMT = "%(asctime)s %(filename)s L%(lineno)d: %(message)s"
DATE_FORMAT = "%y-%m-%d %H:%M:%S"


def get_logger(name=None, log_file=None, log_level=logging.INFO):
    """if log_file is not None, save log into log_file"""
    _logger = logging.getLogger(name)
    logging.basicConfig(format=FMT, datefmt=DATE_FORMAT)
    if log_file is not None:
        log_file_folder = os.path.split(log_file)[0]
        if log_file_folder:
            os.makedirs(log_file_folder, exist_ok=True)
        fh = logging.FileHandler(log_file, "w", encoding="utf-8")
        fh.setFormatter(logging.Formatter(FMT, DATE_FORMAT))
        _logger.addHandler(fh)
    _logger.setLevel(log_level)
    return _logger


logger = get_logger()


def log_df_basic_info(
    df, log_func=None, comments="", full_info=False, partial_col_num=4
):
    """For large df, describe() use much more time, and so skip it for large df!"""
    if log_func is None:
        log_func = logger
    if comments:
        log_func.info(f"comments {comments}")
    log_func.info(f"df.shape {df.shape}")
    columns = df.columns.to_list()
    if full_info:
        log_func.info(f"df.columns {columns}")
        log_func.info(f"df.head()\n{df.head()}")
        log_func.info(f"df.tail()\n{df.tail()}")
        log_func.info(f"df.describe()\n{df.describe()}")
    else:
        log_func.info(f"df.columns [:{partial_col_num}] {columns[:partial_col_num]}")
        log_func.info(f"df.columns [-{partial_col_num}:] {columns[-partial_col_num:]}")
        log_func.info(
            f"df[columns[:{partial_col_num}]].head() {df[columns[:partial_col_num]].head()}"
        )
        log_func.info(
            f"df[columns[-{partial_col_num}:]].tail() {df[columns[-partial_col_num:]].tail()}"
        )
        if len(df) <= 200_000:
            log_func.info(
                f"df[columns[:{partial_col_num}]].describe()\n{df[columns[:partial_col_num]].describe()}"
            )


@contextlib.contextmanager
def timing(msg: str = ""):
    logging.info("Started %s", msg)
    tic = time.time()
    yield
    toc = time.time()
    logging.info("Finished %s in %.3f seconds", msg, toc - tic)


def save_args(args, output_dir=".", with_time_at_filename=False):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    t0 = datetime.now().strftime(DATE_FORMAT)
    if with_time_at_filename:
        out_file = output_dir / f"args-{t0}.txt"
    else:
        out_file = output_dir / "args.txt"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"{t0}\n")
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")


def log_args(args, _logger=None, save_dir=None):
    if _logger is None:
        _logger = logging.getLogger()
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s %(filename)s %(lineno)d: %(message)s",
            datefmt="%y-%m-%d %H:%M",
        )

    args_str = ['args:']
    if not isinstance(args, dict):
        d = vars(args)
    else:
        d = args
    for arg, value in d.items():
        args_str.append(f"{arg}: {value}")
    _logger.info("\n".join(args_str))

    if save_dir is not None:
        _logger.info("Save args to the dir %s", save_dir)
        save_args(args, save_dir)


if __name__ == "__main__":

    @timing()
    def a():
        """ """
        print(1)

    a()
