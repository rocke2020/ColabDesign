import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(""))
warnings.simplefilter(action="ignore", category=FutureWarning)
sys.path.append(os.path.abspath("."))
import random

import jax.numpy as jnp
from jax.lax import dynamic_slice

from alphafold_design.utils.basic import (
   save_current_pdb
)
from colabdesign import clear_mem, mk_afdesign_model
from colabdesign.af.loss import _get_con_loss
from utils_comm.file_util import file_util
from utils_comm.log_util import ic, log_args, logger

parser = argparse.ArgumentParser()
parser.add_argument("--args_file", default="alphafold_design/args/demo_binder.yml")
raw_args = parser.parse_args()
args = file_util.read_yml(raw_args.args_file)
log_args(args, logger)

out_dir = Path(args.root_out_dir) / f"{args.task}-{args.protocol}"
out_dir.mkdir(exist_ok=True, parents=True)


def get_con_loss(
    dgram, dgram_bins, cutoff=None, binary=True, num=1, seqsep=0, offset=None
):
    """convert distogram into contact loss"""
    x = _get_con_loss(dgram, dgram_bins, cutoff, binary)
    a, b = x.shape
    if offset is None:
        mask = jnp.abs(jnp.arange(a)[:, None] - jnp.arange(b)[None, :]) >= seqsep
    else:
        mask = jnp.abs(offset) >= seqsep
    x = jnp.sort(jnp.where(mask, x, jnp.nan))
    k_mask = (jnp.arange(b) < num) * (jnp.isnan(x) == False)
    return jnp.where(k_mask, x, 0.0).sum(-1) / (k_mask.sum(-1) + 1e-8)


def generate_disulfide_pattern(length, disulfide_num=1, min_sep=5):
    disulfide_pattern = []
    positions = list(range(length))
    for n in range(disulfide_num):
        for _ in range(100):  # try 100 time per postion.
            i, j = random.sample(positions, k=2)
            if abs(i - j) <= min_sep:
                continue  # set min loop len.
            positions.remove(i)
            positions.remove(j)
            disulfide_pattern.append((i, j))
            # check
            if _ > 99:
                print("Not find good disulfide_pos! exit....")
                return 0  # not good pose!
            else:
                break
    sequence_pattern = list("X" * length)
    for pair in disulfide_pattern:
        for i in pair:
            sequence_pattern[i] = "C"

    return disulfide_pattern, "".join(sequence_pattern)


def disulfide_loss(inputs, outputs):
    def get_disulfide_loss(dgram, dgram_bins, disulfide_pattern):
        """
        Func: simple disulfide loss, make the contacts < 7.0/7.5A.
        # see: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7316719/
        params: disulfide_pattern: List[(pos1, pos2), (pos3, pos4)...]
        """
        disulfide_loss_value = jnp.array([0.0])
        for pair in disulfide_pattern:
            i, j = pair
            pair_dgram = dynamic_slice(
                dgram, (i, j, 0), (1, 1, len(dgram_bins))
            ) + dynamic_slice(dgram, (j, i, 0), (1, 1, len(dgram_bins)))
            disulfide_loss_value += get_con_loss(
                pair_dgram, dgram_bins, cutoff=7.0, binary=False, num=1
            )
        return disulfide_loss_value.mean()

    # add disulfide loss here:
    dgram_logits = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0, outputs["distogram"]["bin_edges"])
    loss = get_disulfide_loss(
        dgram_logits, dgram_bins, inputs["opt"]["disulfide_pattern"]
    )
    # logger.info(f'{loss = }')
    return {"disulfide": loss}

# disulfide_pattern: [(28, 8)]
# sequence_pattern: XXXXXXXXCXXXXXXXXXXXXXXXXXXXCXXXXXX
L = 35
disulfide_patterns, sequence_pattern = generate_disulfide_pattern(L, 1)
ic(disulfide_patterns, sequence_pattern)
clear_mem()
model = mk_afdesign_model(
    protocol="hallucination",
    loss_callback=disulfide_loss,
    data_dir=args.af_data_path,
)
model.opt["weights"]["disulfide"] = 1.0
model.prep_inputs(length=L)
logger.info(f"{model._len = }")
# weights {'con': 1.0, 'disulfide': 1.0, 'exp_res': 0.0, 'helix': 0.0, 'pae': 0.0, 'plddt': 0.0, 'seq_ent': 0.0}
logger.info(f'{model.opt["weights"] = }')
# set disulfide_pattern sequence.
model.restart(seq=sequence_pattern, add_seq=True, rm_aa="C")
# set disulfide_pattern:
model.opt["disulfide_pattern"] = disulfide_patterns
# reweight con:
model.opt["weights"]["con"] = 0.5
model.design_3stage(50, 50, 10)
save_current_pdb(args, out_dir, model, designed_chain='A')
# af_model.plot_pdb(show_sidechains=True)
logger.info("end")
