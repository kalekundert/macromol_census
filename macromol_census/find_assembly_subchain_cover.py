"""
In each structure, find the minimal set of biological assemblies necessary to 
include every subchain.

Usage:
    mmc_find_assembly_subchain_cover <in:db>

Arguments:
    <in:db>
        The path to a database created by the `mmc_ingest_mmcif` command.

There are some proteins that exist, for example, as both monomers and dimers.  
Structures of such proteins will sometimes contain separate biological 
assemblies for both states.  This script finds these cases and indicates that
the largest assembly is the one that should be used.  The reason for preferring 
larger assemblies is that they have more inter-monomer contacts, which are good 
to include in the dataset.  In my experience, this eliminates 7-8% of the 
dataset.

This is an example of the set-cover problem, which in general is quite 
expensive to solve.  Fortunately, all of the structures in the PDB have few 
enough subchains that the problem can be solved quickly.
"""

import numpy as np
import polars as pl

from scipy.optimize import milp, Bounds, LinearConstraint
from .database_io import open_db, transaction, insert_assembly_subchain_cover
from tqdm import tqdm

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])

    with transaction(db):
        insert_assembly_subchain_covers(db)

def insert_assembly_subchain_covers(db):
    df = db.sql('''\
            SELECT
                struct_id,
                assembly_id,
                subchain_id
            FROM assembly_subchain
            JOIN assembly on assembly.id = assembly_subchain.assembly_id
    ''').pl()
    n = db.sql('SELECT count(*) FROM structure').pl().item()

    for _, assembly_subchain in tqdm(df.group_by(['struct_id']), total=n):
        cover = _find_assembly_subchain_cover(assembly_subchain)
        insert_assembly_subchain_cover(db, cover)

def _find_assembly_subchain_cover(assembly_subchain):
    assembly_i = (
            assembly_subchain
            .select('assembly_id')
            .unique()
            .sort('assembly_id')
            .select(
                pl.int_range(pl.len()).alias('assembly_i'),
                pl.col('assembly_id'),
            )
    )

    subchain_i = (
            assembly_subchain
            .select('subchain_id')
            .unique()
            .sort('subchain_id')
            .select(
                pl.int_range(pl.len()).alias('subchain_i'),
                pl.col('subchain_id'),
            )
    )

    i = (
            assembly_subchain
            .join(assembly_i, on='assembly_id')
            .join(subchain_i, on='subchain_id')
            .select('assembly_i', 'subchain_i')
    )

    # *A* is a matrix that specifies which assemblies contain which subchains.  
    # The structure of the matrix is as follows:
    #
    # - Each column corresponds to an assembly
    # - Each row corresponds to a subchain
    # - Each value is 1 if the assembly contains the subchain, 0 otherwise.

    A = np.zeros((len(subchain_i), len(assembly_i)))
    A[i['subchain_i'], i['assembly_i']] = 1

    res = milp(
            c=np.ones(len(assembly_i)),
            integrality=np.ones(len(assembly_i)),
            bounds=Bounds(lb=0, ub=1),
            constraints=LinearConstraint(A, lb=1),
    )
    assert res.success

    covering_assembly = (
            assembly_i
            .join(
                pl.DataFrame({
                    'assembly_i': np.arange(len(assembly_i)),
                    'select': res.x.astype(int),
                }),
                on='assembly_i',
            )
            .filter(
                pl.col('select') != 0
            )
            .select('assembly_id')
    )

    return covering_assembly


