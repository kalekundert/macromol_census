"""\
Ingest data from validation reports provided by the PDB.

Usage:
    mmc_ingest_validation <in:db> <in:validation-dir>

Arguments:
    <in:db>
        The path to a database created by `mmc_ingest_mmcif`.

    <in:validation-dir>
        The path to a directory containing PDB validation reports, in the 
        `*.cif.gz` format.
"""

import polars as pl

from .working_db import (
        open_db, transaction,
        update_quality_nmr, insert_quality_em, insert_quality_clashscore,
)
from .ingest_mmcif import _extract_dataframe
from .error import add_path_to_ingest_error
from gemmi.cif import read as read_cif
from more_itertools import only
from pathlib import Path
from tqdm import tqdm

def main():
    import docopt

    args = docopt.docopt(__doc__)
    db = open_db(args['<in:db>'])
    val_dir = Path(args['<in:validation-dir>'])
    cif_paths = val_dir.glob('**/*_validation.cif.gz')

    ingest_validation_reports(db, tqdm(list(cif_paths)))

def ingest_validation_reports(db, cif_paths):
    # If the program gets interrupted by some sort of error, there's no easy 
    # way to tell where we left off and to restart from there.  So instead, 
    # wrap the whole program in a single transaction.

    with transaction(db):
        for cif_path in cif_paths:
            ingest_validation_report(db, cif_path)

def ingest_validation_report(db, cif_path):
    with add_path_to_ingest_error(cif_path):
        cif = read_cif(str(cif_path)).sole_block()
        pdb_id = cif.name.lower()

        if n := _extract_nmr_restraints(cif):
            update_quality_nmr(db, pdb_id, num_dist_restraints=n)

        if kw := _extract_em_resolution_q_score(cif):
            insert_quality_em(db, pdb_id, **kw)

        if x := _extract_clashscore(cif):
            insert_quality_clashscore(db, pdb_id, clashscore=x)

def _extract_nmr_restraints(cif):
    restraint_summary = _extract_dataframe(
            cif, 'pdbx_vrpt_restraint_summary',
            required_cols=['description', 'value'],
    )

    if restraint_summary.is_empty():
        return None

    try:
        row = restraint_summary.row(
                by_predicate=pl.col('description') == 'Total distance restraints',
                named=True,
        )
    except pl.exceptions.NoRowsReturnedError:
        return None

    return int(row['value'])

def _extract_em_resolution_q_score(cif):
    cols = [
            'calculated_fsc_resolution_by_cutoff_pt_143',
            'author_provided_fsc_resolution_by_cutoff_pt_143',
            'EMDB_resolution',
            'Q_score',
    ]

    return only(
            _extract_dataframe(
                cif, 'pdbx_vrpt_summary_em',
                optional_cols=cols,
            )
            .lazy()
            # Need to cast before coalescing, because we want to ignore values 
            # that can't be parsed for any reason.
            .select(
                pl.col(*cols).cast(float, strict=False)
            )
            .select(
                resolution_A=pl.coalesce(
                    'calculated_fsc_resolution_by_cutoff_pt_143',
                    'author_provided_fsc_resolution_by_cutoff_pt_143',
                    'EMDB_resolution',
                ),
                q_score='Q_score',
            )
            .collect()
            .to_dicts()
    )

def _extract_clashscore(cif):
    return only(
            _extract_dataframe(
                cif, 'pdbx_vrpt_summary_geometry',
                optional_cols=['clashscore'],
            )
            .get_column('clashscore')
            .cast(float, strict=False)
    )

