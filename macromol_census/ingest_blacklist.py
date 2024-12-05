"""\
Indicate that certain PDB entries should be excluded from the dataset.

Usage:
    mmc_ingest_blacklist <in:db> <in:blacklist>

Arguments:
    <in:db>
        The path to a database created by the `mmc_init` command.

    <in:blacklist>
        A text file containing a single PDB id on each line.  Lines beginning 
        with `#` will be ignored, as will leading/trailing whitespace.

The intended use of this program is to exclude structures that will be used in 
downstream validation/test sets.
"""

import polars as pl
from .database_io import open_db, insert_blacklisted_structures
from pathlib import Path

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])
    ingest_blacklist(db, args['<in:blacklist>'])

def ingest_blacklist(db, txt_path):
    pdb_ids = []

    for line in Path(txt_path).read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            pdb_ids.append(line)

    blacklist = pl.DataFrame(
            {'pdb_id': pdb_ids},
            {'pdb_id': str},
    )
    insert_blacklisted_structures(db, blacklist)



