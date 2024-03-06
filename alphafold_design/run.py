import argparse
import os
import shutil
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(''))
warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append(os.path.abspath("."))
from alphafold_design.utils.basic import (
    check_and_normalize_seq,
    get_all_saved_pdb_seqs,
    get_input_pdb,
    save_all_pdb,
)
from colabdesign import clear_mem, mk_afdesign_model
from utils_comm.file_util import file_util
from utils_comm.log_util import log_args, logger

parser = argparse.ArgumentParser()
parser.add_argument("--args_file", default="alphafold_design/args/demo_binder.yml")
raw_args = parser.parse_args()
args = file_util.read_yml(raw_args.args_file)
log_args(args, logger)

out_dir = Path(args.root_out_dir) / f"{args.task}-{args.protocol}"
out_dir.mkdir(exist_ok=True, parents=True)
target_hotspot = args.target_hotspot
if target_hotspot == "":
    target_hotspot = None

binder_len = args.binder_len
args.binder_seq = check_and_normalize_seq(args.binder_seq)
if args.binder_seq:
    binder_len = len(args.binder_seq)
postfix = f"{args.task}_{args.protocol}-s{args.seed}"


configs = {
    "pdb_filename": get_input_pdb(args.input_pdb_file),
    "chain": args.target_chain,
    "binder_len": binder_len,
    "hotspot": target_hotspot,
    "use_multimer": args.use_multimer,
    "rm_target_seq": args.target_flexible,
}


def loop_design():
    """Use short name for not key information."""
    for _iter in range(args.max_loop_count):
        pdb_file_postfix = f"{postfix}-i{_iter}.pdb"
        logger.info(f"{_iter} loop design")
        design(_iter, args.protocol, pdb_file_postfix)


def design(count, protocol, pdb_file_postfix: str):
    """Run AlphaFold design in a single loop."""
    clear_mem()
    model = mk_afdesign_model(
        protocol=protocol,
        use_multimer=args.use_multimer,
        num_recycles=args.num_recycles,
        recycle_mode=args.recycle_mode,
        data_dir=args.af_data_path,
    )
    model.prep_inputs(**configs, ignore_missing=False)
    # in binder protocol, model._target_len = 85, model._binder_len = 13, model._len = 13
    if count == 0:
        logger.info(f"{model._target_len = }, {model._binder_len = }")
        verbose = 1
    else:
        verbose = 0

    model.restart(seq=args.binder_seq)
    model.set_optimizer(
        optimizer=args.gd_method,
        learning_rate=args.learning_rate,
        norm_seq_grad=args.norm_seq_grad,
    )
    models = model._model_names[: args.num_models]
    flags = {
        "num_recycles": args.num_recycles,
        "models": models,
        "dropout": args.dropout,
        "verbose": verbose,
    }

    if args.optimizer_protocol == "pssm_semigreedy":
        logger.info(f"{args.soft_iters = } {args.hard_iters = }")
        model.design_pssm_semigreedy(args.soft_iters, args.hard_iters, **flags)

    seq = model.get_seqs()[0]
    pre_saved_seqs = get_all_saved_pdb_seqs(out_dir)
    if seq in pre_saved_seqs:
        logger.info(f"{seq = } has been generated and saved before, skip to save.")
        return
    save_all_pdb(seq, args, out_dir, pdb_file_postfix, model)


if __name__ == "__main__":
    # loop_design()
    design(0, args.protocol, f"{postfix}-i0.pdb")
    logger.info("end")
