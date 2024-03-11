import random

import jax.numpy as jnp
from icecream import ic
from jax.lax import dynamic_slice

from colabdesign.af.loss import _get_con_loss

ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


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


def generate_one_disulfide_pattern(length, min_sep=2):
    """Only consider 1 disulfide bond

    Returns:
        disulfide_indexes, [(28, 8)], also with the name "disulfide_patterns"
        sequence_pattern, XXXXXXXXCXXXXXXXXXXXXXXXXXXXCXXXXXX
    """
    positions = list(range(length))
    max_loop_count = 1000
    for _ in range(max_loop_count):  # try max_loop_count time per postion.
        i, j = random.sample(positions, k=2)
        if abs(i - j) >= min_sep:
            disulfide_pattern = (i, j)
            break
    else:
        raise RuntimeError("Not find good disulfide_pos! exit....")
    sequence_pattern = list("X" * length)
    for i in disulfide_pattern:
        sequence_pattern[i] = "C"
    disulfide_indexes = [disulfide_pattern]
    sequence_pattern = "".join(sequence_pattern)
    return disulfide_indexes, sequence_pattern


def generate_disulfide_pattern(length, disulfide_num=1, min_sep=5):
    """raw codes from web"""
    disulfide_patterns = []
    positions = list(range(length))
    for n in range(disulfide_num):
        for _ in range(100):  # try 100 time per postion.
            i, j = random.sample(positions, k=2)
            if abs(i - j) <= min_sep:
                continue  # set min loop len.
            positions.remove(i)
            positions.remove(j)
            disulfide_patterns.append((i, j))
            # check
            if _ > 99:
                print("Not find good disulfide_pos! exit....")
                return 0  # not good pose!
            else:
                break
    sequence_pattern = list("X" * length)
    for pair in disulfide_patterns:
        for i in pair:
            sequence_pattern[i] = "C"

    return disulfide_patterns, "".join(sequence_pattern)


def disulfide_loss(inputs, outputs, cutoff=7):
    def get_disulfide_loss(dgram, dgram_bins, disulfide_pattern):
        """
        Func: simple disulfide loss, make the Cα distance of cysteine contacts < 7.0A.
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
                pair_dgram, dgram_bins, cutoff=cutoff, binary=False, num=1
            )
        return disulfide_loss_value.mean()

    # add disulfide loss here:
    dgram_logits = outputs["distogram"]["logits"]
    dgram_bins = jnp.append(0, outputs["distogram"]["bin_edges"])
    # inputs["opt"]["disulfide_pattern"], e.g., [(28, 8)]
    ic(dgram_logits.shape, dgram_bins.shape)
    loss = get_disulfide_loss(
        dgram_logits, dgram_bins, inputs["opt"]["disulfide_pattern"]
    )
    # logger.info(f'{loss = }')
    return {"disulfide": loss}


if __name__ == "__main__":
    for i in range(100):
        disulfide_patterns, disulfide_seq_pattern = generate_one_disulfide_pattern(6)
        ic(disulfide_patterns, disulfide_seq_pattern)
