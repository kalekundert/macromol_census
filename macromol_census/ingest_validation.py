"""\
Ingest data from validation reports provided by the PDB.

Usage:
    mmc_ingest_validation <in:db> <in:validation-dir>

Arguments:
    <in:db>
        The path to a database created by `mmc_init`, and previously populated 
        by `mmc_ingest_structures`.

    <in:validation-dir>
        The path to a directory containing PDB validation reports, in the 
        `*.cif.gz` format.
"""

import polars as pl
import re

from .database_io import (
        open_db, transaction, select_structure_id,
        insert_nmr_quality, insert_em_quality, insert_clashscore,
)
from .util import read_cif, extract_dataframe
from .error import add_path_to_ingest_error
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
        cif = read_cif(cif_path)
        pdb_id = cif.name.lower()

        # At the time I wrote this code, there were 30 validation reports that 
        # didn't specify a PDB ID, instead just giving the string "BlockName".  
        # In these cases, try to parse the ID from the file name, and give up 
        # if that doesn't work.

        if pdb_id == 'blockname':
            if m := re.match(r'(\w{4})_validation.cif.gz', cif_path.name):
                pdb_id = m.group(1)
            else:
                return

        struct_id = select_structure_id(db, pdb_id)
        source=dict(source='mmcif_pdbx_vrpt')

        if n := _extract_nmr_restraints(cif):
            insert_nmr_quality(db, struct_id, **source, num_dist_restraints=n)

        if kw := _extract_em_resolution_q_score(cif):
            insert_em_quality(db, struct_id, **source, **kw)

        if x := _extract_clashscore(cif):
            insert_clashscore(db, struct_id, **source, clashscore=x)

def _extract_nmr_restraints(cif):
    restraint_summary = extract_dataframe(
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
    return only(
            extract_dataframe(
                cif, 'pdbx_vrpt_summary_em',
                optional_cols=[
                    'EMDB_resolution',
                    'Q_score',
                ],
            )
            .select(
                resolution_A='EMDB_resolution',
                q_score='Q_score',
            )
            .select(
                pl.col('*').cast(float, strict=False)
            )
            .to_dicts()
    )

def _extract_clashscore(cif):
    return only(
            extract_dataframe(
                cif, 'pdbx_vrpt_summary_geometry',
                optional_cols=['clashscore'],
            )
            .get_column('clashscore')
            .cast(float, strict=False)
            .replace(-1, None)
    )

