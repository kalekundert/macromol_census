"""\
Identify and ignore ligands that are often bound non-specifically.

Usage:
    mmc_ingest_nonspecific_ligands <in:db> <in:ligands> [-w <mwco>]

Arguments:
    <in:db>
        The path to a database created by the `mmc_init` command.

    <in:ligands>
        A text file containing a single PDB component identifier (usually 1-3 
        characters) on each line.

Options:
    -w --mwco <Da>                      [default: 0]
        A molecular weight cutoff, in daltons.  Molecules smaller than this 
        will be considered nonspecific.

The non-specific ligands identified here will not be considered when choosing 
assemblies with unique combinations of entities to include in the dataset.
"""

import polars as pl
from .database_io import open_db, insert_nonspecific_ligands

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])
    ingest_nonspecific_ligands(db, args['<in:ligands>'])
    ignore_low_weight_ligands(db, args['--mwco'])

def ingest_nonspecific_ligands(db, ignore_path):
    ignore = pl.read_csv(
            ignore_path,
            has_header=False,
            new_columns=['pdb_comp_id'],
    )
    insert_nonspecific_ligands(db, ignore)

def ignore_low_weight_ligands(db, mwco):
    ignore = db.sql('''\
            SELECT entity_monomer.pdb_comp_id
            FROM entity_monomer
            JOIN entity ON entity.id = entity_monomer.entity_id
            WHERE entity.formula_weight_Da < ?
    ''', params=[mwco])
    insert_nonspecific_ligands(db, ignore)


