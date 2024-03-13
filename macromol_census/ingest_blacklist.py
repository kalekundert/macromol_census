"""\
Indicate that certain PDB entries should be excluded from the dataset.

Usage:
    mmc_ingest_blacklist <in:db> <in:blacklist>

Arguments:
    <in:db>
        The path to a database created by the `mmc_init` command.

    <in:blacklist>
        A text file containing a single PDB id on each line.

The intended use of this program is to exclude structures that will be used in 
downstream validation/test sets.
"""

import polars as pl
from .database_io import open_db, insert_blacklisted_structures

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
    insert_blacklisted_structures(db, blacklist)



