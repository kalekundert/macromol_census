"""
Indicate that certain PDB entries should be excluded from the dataset.

Usage:
    mmc_ingest_blacklist <in:db> <in:blacklist>

Arguments:
    <in:db>
        The path to a database created by the `mmc_ingest_mmcif` command.

    <in:blacklist>
        A test file containing a single PDB id on each line.

The intended use of this program is to exclude models that will be used in 
downstream validation/test sets.
"""

import polars as pl
from .working_db import open_db, insert_model_blacklist

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])
    ingest_blacklist(db, args['<in:blacklist>'])

def ingest_blacklist(db, csv_path):
    blacklist = pl.read_csv(
            csv_path,
            has_header=False,
            dtypes={'pdb_id': str},
    )
    insert_model_blacklist(db, blacklist)



