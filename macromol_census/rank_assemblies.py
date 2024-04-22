"""
Determine the order in which the assemblies within each structure will be 
considered.

Usage:
    mmc_find_assembly_subchain_cover <in:db>

Arguments:
    <in:db>
        The path to a database created by the `mmc_ingest_mmcif` command.

Note that the rankings produced by this script only apply to the assemblies 
within a single structure.  Assemblies from different structures are not 
ranked; see the `mmc_rank_structures` for that.  That said, ranking the 
assemblies within each structure is important.  Assemblies that are considered 
first are much more likely to end up in the final dataset, because all of the 
assemblies are likely composed of the same molecular entities, so every 
assembly but the first will be considered redundant.  In general, larger 
assemblies are preferred over smaller ones.  The idea is that larger assemblies 
have more inter-monomer contacts, which are good to include in the dataset. 
"""

import numpy as np
import polars as pl

from scipy.optimize import milp, Bounds, LinearConstraint
from .database_io import open_db, insert_assembly_ranks
from .util import tquiet
from tqdm import tqdm

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])

    ranks = rank_assemblies(db, tqdm)
    insert_assembly_ranks(db, ranks)

def rank_assemblies(db, progress_factory=tquiet):
    assemblies = db.sql('''\
            SELECT
                assembly.struct_id AS struct_id,
                assembly.id AS assembly_id,
                assembly_subchain.subchain_ids AS subchain_ids,
                assembly.type AS type,
                assembly.polymer_count AS polymer_count
            FROM assembly
            JOIN (
                SELECT
                    assembly_id,
                    list(subchain_id ORDER BY subchain_id) AS subchain_ids
                FROM assembly_subchain
                GROUP BY assembly_id
            ) AS assembly_subchain
            ON assembly.id = assembly_subchain.assembly_id
    ''').pl()

    n = assemblies.n_unique('struct_id')
    progress = progress_factory(total=n)

    def filter_subchain_cover(df):
        progress.update(1)
        cover = find_assembly_subchain_cover(
                df
                .explode('subchain_ids')
                .rename({'subchain_ids': 'subchain_id'})
        )
        return df.filter(pl.col('assembly_id').is_in(cover))


    return (
            assemblies

            # Only keep assemblies that are "biologically relevant" (see #1):
            .filter(
                pl.col('type').is_in([
                    'representative helical assembly',
                    'complete point assembly',
                    'complete icosahedral assembly',
                    'software_defined_assembly',
                    'author_defined_assembly',
                    'author_and_software_defined_assembly',
                ]),
            )

            # Only keep the largest assembly in each group with all the same 
            # subchains:
            .group_by('struct_id', 'subchain_ids')
            .agg(pl.all().sort_by('polymer_count').last())

            # Only keep the minimum number of assemblies needed to include 
            # every subchain:
            .group_by('struct_id')
            .map_groups(filter_subchain_cover)

            # Rank remaining assemblies by size, then by order of appearance in 
            # the mmCIF file:
            .select(
                pl.col('assembly_id')
                    .sort_by(
                        'polymer_count', 'assembly_id',
                        descending=[True, False],
                    )
                    .over('struct_id'),
                (pl.int_range(pl.len()) + 1).over('struct_id').alias('rank'),
            )
    )
                
def find_assembly_subchain_cover(assembly_subchain):
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


