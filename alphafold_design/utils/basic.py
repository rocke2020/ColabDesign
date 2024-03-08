import os
import re
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy import ndarray

sys.path.append(os.path.abspath("."))
from alphafold_design.utils.plot import save_plddts_img
from colabdesign import mk_afdesign_model
from utils_comm.file_util import file_util
from utils_comm.log_util import logger


@dataclass
class AFDesignScores:
    """
    all_plddts: plddts of the whole sequences of all models, 2D array of shape (num_models, L)
        containing pLDDT predicted scores
    """
    protocol: str
    designed_part_plddt: float
    sorted_designed_part_plddts: ndarray
    all_plddts: ndarray
    i_ptm: float = 0.0
    ptm: float = 0.0
    complex_score: float = 0.0
    complex_scores_sorted_by_plddt: ndarray | None = None


non_upper_alpha = re.compile("[^A-Z]")


def calc_plddts(self, get_best=True):
    """
    aux.keys: 'aatype', 'atom_mask', 'atom_positions', 'cmap', 'grad', 'i_cmap', 'i_ptm',
        'loss', 'losses', 'num_recycles', 'pae', 'plddt', 'prev', 'ptm', 'residue_index',
        'seq', 'seq_pseudo'
    """
    # "aux" in self._tmp["best"] = True
    _aux = (
        self._tmp["best"]["aux"]
        if (get_best and "aux" in self._tmp["best"])
        else self.aux
    )
    aux = _aux["all"]
    all_plddts = 100 * aux["plddt"]
    plddts_of_designed_part = (
        all_plddts[:, self._target_len :]
        if self.protocol == "binder"
        else all_plddts[:, : self._len]
    )

    mean_designed_part_plddts = plddts_of_designed_part.mean(axis=1)
    ind = np.argsort(mean_designed_part_plddts)[::-1]
    sorted_designed_part_plddts = mean_designed_part_plddts[ind]
    designed_part_plddt = sorted_designed_part_plddts.mean()
    i_ptm, ptm, complex_score, complex_scores_sorted_by_plddt = 0.0, 0.0, 0.0, None
    if self.protocol == "binder":
        i_ptm = aux["i_ptm"]
        ptm = aux["ptm"]
        complex_scores = i_ptm * 0.8 + ptm * 0.2
        complex_scores_sorted_by_plddt = complex_scores[ind]
        complex_score = complex_scores_sorted_by_plddt.mean()
        i_ptm = i_ptm.mean()
        ptm = ptm.mean()
    results = AFDesignScores(
        self.protocol,
        designed_part_plddt,
        sorted_designed_part_plddts,
        all_plddts,
        i_ptm,
        ptm,
        complex_score,
        complex_scores_sorted_by_plddt,
    )
    best_model_complex_scores = (
        complex_scores_sorted_by_plddt[0]
        if complex_scores_sorted_by_plddt is not None
        else 0.0
    )
    logger.info(
        f"{self.protocol} {designed_part_plddt = } {complex_score = }, "
        f"best_model_plddt by plddt {sorted_designed_part_plddts[0]}, "
        f"best_model_complex_scores {best_model_complex_scores}"
    )
    return results


def save_best_model_pdb(
    all_models_pdb_file: Path,
    save_file: Path | None = None,
    designed_chain: str = "B",
    verbose=0,
):
    """default saved pdb have 5 models, only for simple pdb from AF design.

    MODEL       1
    ATOM      1  N   ILE A   1     156.903 134.037 170.302  1.00 73.31           N
    ATOM      2  CA  ILE A   1     157.520 133.667 169.033  1.00 73.31           C
    ATOM      3  C   ILE A   1     158.387 134.817 168.526  1.00 73.31           C
    ATOM      4  CB  ILE A   1     156.458 133.290 167.977  1.00 73.31           C
    ATOM      5  O   ILE A   1     158.080 135.987 168.767  1.00 73.31           O
    ATOM      6  CG1 ILE A   1     155.625 132.097 168.459  1.00 73.31           C
    ATOM      7  CG2 ILE A   1     157.120 132.988 166.629  1.00 73.31           C
    ATOM      8  CD1 ILE A   1     154.459 131.746 167.544  1.00 73.31           C
    ATOM      9  N   ALA A   2     159.514 134.407 167.983  1.00 73.10           N

    ENDMDL
    MODEL       2
    ATOM      1  N   ILE A   1     156.824 134.131 170.401  1.00 76.25           N

    ATOM   2193  CD  PRO B   6     148.383 139.488 173.043  1.00 71.73           C
    ENDMDL
    END

    """
    models_lines = []
    with open(all_models_pdb_file, "r", encoding="utf-8") as f:
        model_lines = []
        model_plddts = defaultdict(list)
        for line in f:
            if line.startswith("MODEL"):
                if model_lines:
                    chain_plddts = [i[1] for i in model_plddts[designed_chain]]
                    designed_chain_plddt = sum(chain_plddts) / len(chain_plddts)
                    models_lines.append((model_lines.copy(), designed_chain_plddt))
                    model_lines.clear()
                    model_plddts = defaultdict(list)
                model_lines.append(line)
            elif line.startswith("ATOM"):
                model_lines.append(line)
                items = line.split()
                chain = items[4]
                index = items[5]
                plddt = float(items[-2])
                chain_list = model_plddts[chain]
                if chain_list:
                    if chain_list[-1][0] != index:
                        chain_list.append([index, plddt])
                else:
                    chain_list.append([index, plddt])
            elif line.startswith("ENDMDL"):
                model_lines.append(line)
            elif line.startswith("END"):
                chain_plddts = [i[1] for i in model_plddts[designed_chain]]
                designed_chain_plddt = sum(chain_plddts) / len(chain_plddts)
                models_lines.append((model_lines, designed_chain_plddt))

    ranked_models = sorted(models_lines, key=lambda x: x[1], reverse=True)
    designed_chain_plddts_all_models = [i[1] for i in ranked_models]
    if verbose:
        logger.info(f"{all_models_pdb_file.name} {designed_chain_plddts_all_models = }")
        average_plddt = sum(designed_chain_plddts_all_models) / len(ranked_models)
        logger.info(f"{designed_chain = } {average_plddt = }")

    if not save_file:
        save_dir = all_models_pdb_file.parent / "best_model_pdb"
        save_dir.mkdir(exist_ok=True)
        save_file = save_dir / all_models_pdb_file.name
    with open(save_file, "w", encoding="utf-8") as f:
        model_lines, designed_chain_plddt = ranked_models[0]
        logger.info(f"Save the top 1 model pdb with {designed_chain_plddt = }")
        for line in model_lines:
            f.write(line)
        f.write("END\n")


def save_best_model_pdbs(all_models_pdb_dir, verbose=1):
    """ tmp usage, new version will auto save best model pdb and json file."""
    all_models_pdb_dir = Path(all_models_pdb_dir)
    saved_pdbs = list(all_models_pdb_dir.glob("*.pdb"))
    logger.info(f"{len(saved_pdbs) = }")
    for all_models_pdb_file in saved_pdbs:
        save_best_model_pdb(all_models_pdb_file, verbose=verbose)
    best_model_dir = all_models_pdb_dir / "best_model_pdb"
    best_model_dir.mkdir(exist_ok=True)
    for json_file in all_models_pdb_dir.glob("*.json"):
        shutil.move(json_file, best_model_dir / json_file.name)


def get_all_saved_pdb_seqs(pdb_out_parent_dir, sub_dir="all_models_pdb"):
    """sub_dir: all_models_pdb best_model_pdb/correct_binded_pdbs"""
    pdb_out_parent_dir = Path(pdb_out_parent_dir)
    pdb_dir = pdb_out_parent_dir / sub_dir
    pdb_dir.mkdir(exist_ok=True, parents=True)
    saved_seqs = []
    for pdb_file in pdb_dir.glob("*.pdb"):
        items = pdb_file.stem.split("-")
        for i, item in enumerate(items):
            if item == "seq" and i < len(items) - 1:
                if items[i + 1] in saved_seqs:
                    logger.info(f"{pdb_file.name} seq is saved before in this dir.")
                    pdb_file.unlink()
                else:
                    saved_seqs.append(items[i + 1])
    # logger.info(f"{len(saved_seqs) = }")
    return saved_seqs


def get_input_pdb(pdb_code):
    if os.path.isfile(pdb_code):
        return pdb_code
    if len(pdb_code) == 4:
        file = f"{pdb_code}.pdb"
        if not Path(file).exists():
            os.system(f"wget -qnc https://files.rcsb.org/view/{pdb_code}.pdb")
        return file
    os.system(
        f"wget -qnc https://alphafold.ebi.ac.uk/files/AF-{pdb_code}-F1-model_v3.pdb"
    )
    return f"AF-{pdb_code}-F1-model_v3.pdb"


def check_and_normalize_seq(seq=""):
    """ If seq is empty, convert to return None, otherwise raise error in design."""
    if non_upper_alpha.search(seq):
        raise ValueError(f"{seq = } has not A-Z chars")
    if not seq:
        seq = None
    return seq


def save_current_pdb(
    args,
    out_dir: Path,
    model: mk_afdesign_model,
    designed_chain: str,
    generated_seq: str | None = None,
    pdb_file_postfix: str = ".pdb",
):
    """just simple save func, not filter low confident pdb."""
    results = calc_plddts(model)
    # Unique and most important info at head, that's seq here.
    all_models_dir = out_dir / "all_models_pdb"
    all_models_dir.mkdir(exist_ok=True, parents=True)
    if not generated_seq:
        generated_seq = model.get_seqs()[0]
    out_pdb_filename = (
        f"seq-{generated_seq}-plddt{results.designed_part_plddt:.2f}-"
        f"complexScore{results.complex_score:.2f}-{pdb_file_postfix}"
    )
    all_models_pdb_file = all_models_dir / out_pdb_filename
    model.save_pdb(all_models_pdb_file)
    img_file = all_models_pdb_file.with_suffix(".png")
    save_plddts_img(img_file, results.all_plddts, model._lengths)

    best_model_dir = out_dir / "best_model_pdb"
    best_model_dir.mkdir(exist_ok=True)
    best_model_pdb_file = best_model_dir / out_pdb_filename
    save_best_model_pdb(all_models_pdb_file, best_model_pdb_file, designed_chain)
    result_file = best_model_dir / f"{best_model_pdb_file.stem}.json"
    result_dict = vars(results)
    result_dict["Sequence"] = generated_seq
    result_dict["input_binder_seq"] = args.get('binder_seq', None)
    result_dict["args"] = args
    result_dict.pop("all_plddts")

    file_util.write_json(result_dict, result_file)


if __name__ == "__main__":
    # normalize_seq(seq="")
    # save_best_model_pdb(
    # "/mnt/nas1/alphafold-design/output/8inr-MC5R-binder/8inr-MC5R_cyclic_binder-seed1-iter0-seq-IFSPNN-plddt72.20-complexScore0.82.pdb"
    # )
    # save_best_model_pdbs(
    #     "/mnt/nas1/alphafold-design/output/8inr-MC5R-binder", verbose=0
    # )
    get_all_saved_pdb_seqs('/mnt/nas1/alphafold-design/output/8inr-MC5R-binder/')
