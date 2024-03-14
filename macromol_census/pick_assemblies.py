"""
Pick a non-redundant set of biological assemblies.

Usage:
    mmc_pick_assemblies <in:db>

Arguments:
    <in:db>
        A database created by the various `mmc_ingest_*` commands.
"""

import polars as pl
from .database_io import open_db, select_assemblies, select_assembly_subchains
from itertools import combinations
from tqdm import tqdm

def main():
    import docopt

    args = docopt.docopt(__doc__)
    db = open_db(args['<in:db>'])

    pick_assemblies(db)

def pick_assemblies(db):
    # KBK: Below is an outline of the original algorithm I planned.  The final 
    # version ended up a little different, but I haven't updated the notes yet.  
    # I'll probably want to write some docstrings first.

    # Get relevant assemblies:
    #
    # - Remove any that are blacklisted
    # - Remove any that are worse than 10Å resolution
    # - Remove any that aren't need to cover the subchains
    #
    # - Sort by "quality", i.e. the following metrics, in order:
    #   - resolution, if better than 4Å, rounded to nearest 0.1
    #   - clashscore, rounded to nearest 0.2
    #   - NMR restraints
    #   - R free
    #   - Q score
    #   - date
    #   - PDB id
    #
    # Get relevant subchains:
    #
    # - Remove polymers below MW cutoff
    # - Remove non-polymers that are non-specific or non-biological.
    #
    # Indicate which subchains are "equivalent":
    #
    # - Polymers: entities in same sequence cluster
    # - Non-polymers: entities have same name, or maybe InChI key.
    #
    # Choose assemblies to include:
    #
    # - Greedy first pass:
    #   - Start with highest "quality" assembly.
    #   - Find new subchains/subchain pairs in this assembly.
    #   - If none: continue to next assembly
    #   - Else: record which chains/chain pairs "belong" to this assembly
    #   - Advance to next highest "quality" assembly.
    # 
    # - Clean up:
    #   - Start with highest "quality" assembly in dataset.
    #   - Remove this assembly if another one contains all the same 
    #     chains/chain pairs.
    #   - Advance to next highest "quality" assembly.

    included_clusters = set()
    included_cluster_pairs = set()

    relevant_subchains = _select_relevant_subchains(db)
    relevant_assemblies = _select_relevant_assemblies(db, relevant_subchains)
    assembly_subchains = db.sql('''\
            SELECT
                structure.rank AS rank,
                assembly.id AS assembly_id,
                relevant_subchains.subchain_id AS subchain_id,
                relevant_subchains.cluster_id AS cluster_id
            FROM relevant_assemblies
            JOIN assembly ON assembly.id = relevant_assemblies.assembly_id
            JOIN assembly_subchain USING (assembly_id)
            JOIN relevant_subchains USING (subchain_id)
            JOIN structure ON structure.id = assembly.struct_id
            ORDER BY rank, assembly_id, subchain_id
    ''').pl()
    n = assembly_subchains.n_unique('assembly_id')

    # This isn't guaranteed to free the memory used by these data frames, but 
    # it at least makes it possible for that to happen.
    del relevant_subchains
    del relevant_assemblies

    for _, assembly_i in tqdm(
            assembly_subchains.group_by(['assembly_id'], maintain_order=True),
            total=n,
    ):
        nonredundant = []
        nonredundant_pairs = []

        subchains_i = assembly_i.select('subchain_id', 'cluster_id')

        for subchain_id, cluster_id in subchains_i.iter_rows():
            if cluster_id not in included_clusters:
                included_clusters.add(cluster_id)
                nonredundant.append(subchain_id)

        for (subchain_j, cluster_j), (subchain_k, cluster_k) in \
                combinations(subchains_i.iter_rows(), r=2):
            pair = frozenset([cluster_j, cluster_k])
            if pair not in included_cluster_pairs:
                included_cluster_pairs.add(pair)
                nonredundant_pairs.append(sorted([subchain_j, subchain_k]))

        nonredundant_df = pl.DataFrame(
                nonredundant,
                schema=['subchain_id'],
        )
        nonredundant_pairs_df = pl.DataFrame(
                nonredundant_pairs,
                schema=['subchain_id_1', 'subchain_id_2'],
                orient='row',
        )

        db.sql('''\
                INSERT INTO nonredundant (subchain_id)
                SELECT subchain_id FROM nonredundant_df;

                INSERT INTO nonredundant_pair (subchain_id_1, subchain_id_2)
                SELECT subchain_id_1, subchain_id_2 FROM nonredundant_pairs_df;
        ''')

def _select_relevant_subchains(db):
    entity_cluster_some = db.sql('''\
            SELECT 
                entity.id AS entity_id,
                entity_cluster.cluster_id AS cluster_id
            FROM entity
            LEFT JOIN entity_cluster ON entity.id = entity_cluster.entity_id
    ''').pl()


    # Entities that have null cluster ids are in their own clusters (or in 
    # other words, aren't clustered with anything).  We want entities in 
    # different clusters to have different cluster id numbers, so we need to 
    # replace null values with unique ids.

    singleton_cluster_start = entity_cluster_some['cluster_id'].max() + 1

    entity_cluster_all = (
            entity_cluster_some
            .with_columns(
                cluster_id=pl.coalesce(
                    'cluster_id',
                    pl.lit(singleton_cluster_start)
                    + pl.int_range(pl.len()).over('cluster_id')
                )
            )
    )

    return db.sql('''\
            SELECT 
                subchain.id AS subchain_id,
                entity_cluster_all.cluster_id AS cluster_id
            FROM subchain
            JOIN entity_cluster_all USING (entity_id)
            ANTI JOIN entity_ignore USING (entity_id)
    ''').pl()

def _select_relevant_assemblies(db, subchain_cluster):
    assembly_cluster = db.sql('''\
            SELECT
                assembly_subchain.assembly_id AS assembly_id,
                subchain_cluster.cluster_id AS cluster_id
            FROM assembly_subchain
            JOIN subchain_cluster USING (subchain_id)
    ''')

    cluster_blacklist = db.sql('''\
            SELECT DISTINCT
                cluster_id
            FROM structure_blacklist
            JOIN assembly USING (struct_id)
            JOIN assembly_cluster ON assembly.id = assembly_cluster.assembly_id
    ''')

    assembly_blacklist = db.sql('''\
            SELECT DISTINCT
                assembly_cluster.assembly_id AS assembly_id
            FROM cluster_blacklist
            JOIN assembly_cluster USING (cluster_id)
    ''')

    return db.sql('''\
            SELECT
                assembly_subchain_cover.assembly_id AS assembly_id
            FROM assembly_subchain_cover
            ANTI JOIN assembly_blacklist USING (assembly_id)
    ''').pl()

