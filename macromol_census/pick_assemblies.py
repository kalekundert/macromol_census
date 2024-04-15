"""
Pick a non-redundant set of biological assemblies.

Usage:
    mmc_pick_assemblies <in:db>

Arguments:
    <in:db>
        A database created by the various `mmc_ingest_*` commands.
"""

import polars as pl
import pickle

from .database_io import open_db, transaction
from .util import tquiet
from dataclasses import dataclass
from itertools import combinations
from more_itertools import flatten
from functools import cached_property
from tqdm import tqdm

from typing import TypeAlias
from collections.abc import Iterable

PdbId: TypeAlias = str

class Visitor:
    """
    The default implementation doesn't actually do any filtering.
    """

    def propose(self, assembly):
        raise NotImplementedError

    def accept(self, candidates, memento):
        raise NotImplementedError

class Assembly:

    def __init__(self, db, assembly_id, subchain_clusters):
        # The data used to initialize these objects are basically whatever is 
        # conveniently available to the main loop.  Anything not conveniently 
        # available will be queried on-demand.  Note that the constructor is 
        # not considered part of the public API, and is subject to change at 
        # any time.
        self._db = db
        self._assembly_id = assembly_id
        self._subchain_clusters = subchain_clusters

    def __repr__(self):
        return f'<Assembly structure={self.struct_pdb_id} assembly={self.assembly_pdb_id}>'

    @cached_property
    def struct_pdb_id(self):
        df = self._db.sql('''\
                SELECT structure.pdb_id
                FROM assembly
                JOIN structure ON structure.id = assembly.struct_id
                WHERE assembly.id = ?
        ''', params=[self._assembly_id]).pl()
        return df['pdb_id'].item()

    @cached_property
    def model_pdb_ids(self):
        df = self._db.sql('''\
                SELECT model.pdb_id
                FROM assembly
                JOIN structure ON structure.id = assembly.struct_id
                JOIN model ON structure.id = model.struct_id
                WHERE assembly.id = ?
        ''', params=[self._assembly_id]).pl()
        return df['pdb_id'].to_list()

    @cached_property
    def assembly_pdb_id(self):
        df = self._db.sql(
                'SELECT pdb_id FROM assembly WHERE id = ?',
                params=[self._assembly_id],
        ).pl()
        return df['pdb_id'].item()

    @cached_property
    def subchain_pdb_ids(self):
        return self._subchain_clusters['subchain_pdb_id'].to_list()

class Memento:

    def __init__(self):
        self._assembly_id = None
        self._accepted_clusters = set()
        self._accepted_cluster_pairs = set()

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)

@dataclass
class Candidate:
    subchains: Iterable[PdbId] = frozenset()
    subchain_pairs: Iterable[tuple[PdbId, PdbId]] = frozenset()

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return hash(id(self))

def main():
    import docopt

    args = docopt.docopt(__doc__)
    db = open_db(args['<in:db>'])

    with transaction(db):
        pick_assemblies(db, progress_factory=tqdm)

def pick_assemblies(db, progress_factory=tquiet):
    nonredundant = []
    nonredundant_pairs = []

    class PickVisitor(Visitor):
        _subchain_col = 'subchain_id'

        def propose(self, assembly):
            # Note that we're accessing private members of the Assembly class 
            # here.  It's best to think of these members as being "module 
            # private"; i.e. accessible within this module, but not outside of 
            # it.
            # 
            # The idea is that the primary keys used database are internal 
            # implementation details and should not be revealed to the outside 
            # world.  Instead, when necessary, the outside world should be 
            # given the identifiers that are used in the PDB (which differ from 
            # the primary keys in that they aren't globally unique).
            #
            # This function needs access to the primary keys, in order to 
            # record the pick assemblies to the database.  This need doesn't 
            # violate any conventions, because this function isn't part of "the 
            # outside world".  However, because the Assembly class can't make 
            # this information public, it means that we need the concept of 
            # "module private" information.
            subchain_ids = sorted(assembly._subchain_clusters['subchain_id'])

            for subchain_id in subchain_ids:
                yield Candidate(subchains=[subchain_id])

            for subchain_pair in combinations(subchain_ids, r=2):
                yield Candidate(subchain_pairs=[subchain_pair])

        def accept(self, candidates, _):
            nonredundant.extend(
                    sorted(flatten(x.subchains for x in candidates))
            )
            nonredundant_pairs.extend(
                    sorted(flatten(x.subchain_pairs for x in candidates))
            )

    visit_assemblies(db, PickVisitor(), progress_factory=progress_factory)

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

def visit_assemblies(db, visitor, *, memento=None, progress_factory=tquiet):
    # KBK: Below is an outline of the original algorithm I planned.  The final 
    # version ended up a little different, but I haven't updated the notes yet.  

    """
    Get relevant assemblies:
    
    - Remove any that are blacklisted
    - Remove any that are worse than 10Å resolution
    - Remove any that aren't need to cover the subchains
    
    - Sort by "quality", i.e. the following metrics, in order:
      - resolution, if better than 4Å, rounded to nearest 0.1
      - clashscore, rounded to nearest 0.2
      - NMR restraints
      - R free
      - Q score
      - date
      - PDB id
    
    Get relevant subchains:
    
    - Remove polymers below MW cutoff
    - Remove non-polymers that are non-specific or non-biological.
    
    Indicate which subchains are "equivalent":
    
    - Polymers: entities in same sequence cluster
    - Non-polymers: entities have same name, or maybe InChI key.
    
    Choose assemblies to include:
    
    - Greedy first pass:
      - Start with highest "quality" assembly.
      - Find new subchains/subchain pairs in this assembly.
      - If none: continue to next assembly
      - Else: record which chains/chain pairs "belong" to this assembly
      - Advance to next highest "quality" assembly.
    
    - Clean up:
      - Start with highest "quality" assembly in dataset.
      - Remove this assembly if another one contains all the same 
        chains/chain pairs.
      - Advance to next highest "quality" assembly.
    """

    if memento is None:
        memento = Memento()

    relevant_subchains = _select_relevant_subchains(db)
    relevant_assemblies = _select_relevant_assemblies(db, relevant_subchains)
    assembly_subchains = db.sql('''\
            SELECT
                structure.rank AS rank,
                assembly.id AS assembly_id,
                subchain.chain_id AS chain_id,
                subchain.id AS subchain_id,
                subchain.pdb_id AS subchain_pdb_id,
                relevant_subchains.cluster_id AS cluster_id
            FROM relevant_assemblies
            JOIN assembly ON assembly.id = relevant_assemblies.assembly_id
            JOIN assembly_subchain USING (assembly_id)
            JOIN structure ON structure.id = assembly.struct_id
            JOIN subchain ON subchain.id = assembly_subchain.subchain_id
            JOIN relevant_subchains USING (subchain_id)
            ORDER BY rank, assembly_id, chain_id, subchain_id
    ''').pl()

    # This isn't guaranteed to free the memory used by these data frames, but 
    # it at least makes it possible for that to happen.
    del relevant_subchains
    del relevant_assemblies

    if (last_assembly_id := memento._assembly_id) is not None:
        last_rank = _select_assembly_rank(db, last_assembly_id)
        assembly_subchains = (
                assembly_subchains
                .filter(
                    (pl.col('rank') >= last_rank) &
                    (pl.col('assembly_id') > last_assembly_id)
                )
        )

    n = assembly_subchains.n_unique('assembly_id')
    progress = progress_factory(total=n)
    subchain_col = getattr(visitor, '_subchain_col', 'subchain_pdb_id')

    for (assembly_id,), assembly_subchains_i in (
            assembly_subchains.group_by(['assembly_id'], maintain_order=True)
    ):
        progress.update()

        assembly = Assembly(db, assembly_id, assembly_subchains_i)
        candidates = list(visitor.propose(assembly))

        cluster_map = dict(
                assembly_subchains_i
                .select(subchain_col, 'cluster_id')
                .iter_rows()
        )
        chain_map = dict(
                assembly_subchains_i
                .select(subchain_col, 'chain_id')
                .iter_rows()
        )
        accepted_candidates = set()

        _accept_nonredundant_subchains(
                candidates,
                cluster_map,
                chain_map,
                accepted_candidates,
                memento._accepted_clusters,
        )
        _accept_nonredundant_subchain_pairs(
                candidates,
                cluster_map,
                chain_map,
                accepted_candidates,
                memento._accepted_cluster_pairs,
        )

        memento._assembly_id = assembly_id
        visitor.accept(accepted_candidates, memento)


def _select_relevant_subchains(db):
    """
    Return all of the subchains eligible to include in the dataset, along with 
    the cluster that each belongs to.

    Returns:
        A dataframe with two columns:

        - ``subchain_id``: A reference to the ``id`` column of the ``subchain`` 
          table.  Subchains that should be excluded from the final dataset 
          (e.g. non-specific ligands) will be excluded from this dataframe.  
          Generally speaking, though, this dataframe will contain most of the 
          subchains provided by the user.  Note that not all of the subchains 
          returned here will necessarily end up in the final dataset; there are 
          subsequent filtering steps.  But any subchains not returned here will 
          not be in the final dataset.

        - ``cluster_id``: The id number of the cluster that this subchain 
          belongs to.  Clusters are determined by the entity the subchain 
          represents.  This column will not have any null values.
    """
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

    singleton_cluster_start = (entity_cluster_some['cluster_id'].max() or 0) + 1

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
    """
    Return all of the assemblies eligible to include in the dataset.

    Returns:
        A dataframe with one column:

        - `assembly_id``: A reference to the ``id`` column of the ``assembly`` 
          table.

    An assembly is deemed eligible to include in the dataset if:

    - It does not appear in a blacklisted structure.
    - It does not share a subchain cluster with any blacklisted assemblies.
    - It is part of the "assembly cover", i.e. the minimal set of assemblies 
      needed to include every subchain.
    - It has a resolution below 10Å.  Assemblies that don't have a resolution 
      at all (e.g. from NMR structures) are not excluded by this criterion.  If 
      the assembly has multiple resolutions, only the lowest is considered.

    Not all of the assemblies returned by 
    this function will necessarily end up in the final dataset.  They 
    have not yet been filtered for redundancy.  However, any assemblies 
    not returned by this function will not be included in the final 
    dataset.
    """
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

    assembly_low_res = db.sql('''\
            SELECT *
            FROM (
                SELECT
                    assembly.id AS assembly_id,
                    least(
                        min(quality_xtal.resolution_A),
                        min(quality_em.resolution_A)
                    ) AS resolution_A
                FROM assembly
                LEFT JOIN quality_xtal USING (struct_id)
                LEFT JOIN quality_em USING (struct_id)
                GROUP BY assembly.id
            )
            WHERE resolution_A >= 10
    ''')

    return db.sql('''\
            SELECT
                assembly_subchain_cover.assembly_id AS assembly_id
            FROM assembly_subchain_cover
            ANTI JOIN assembly_blacklist USING (assembly_id)
            ANTI JOIN assembly_low_res USING (assembly_id)
    ''').pl()

def _select_assembly_rank(db, assembly_id):
    df = db.sql('''\
            SELECT DISTINCT structure.rank AS rank,
            FROM structure
            JOIN assembly ON structure.id = assembly.struct_id
            WHERE assembly.id = ?
    ''', params=[assembly_id]).pl()
    return df['rank'].item()

def _accept_nonredundant_subchains(
        candidates: list[Candidate],
        cluster_map: dict[str, int],
        chain_map: dict[str, int],
        accepted_candidates: set[Candidate],
        accepted_clusters: set[int],
):
    groups = {}
    for candidate in candidates:
        for k in candidate.subchains:
            groups.setdefault(k, []).append(candidate)

    def get_priority(subchain):
        prevalence = -len(groups[subchain])
        return prevalence, chain_map[subchain], subchain

    for subchain in sorted(groups, key=get_priority):
        cluster = cluster_map[subchain]
        if cluster not in accepted_clusters:
            accepted_candidates.update(groups[subchain])
            accepted_clusters.add(cluster)

def _accept_nonredundant_subchain_pairs(
        candidates: list[Candidate],
        cluster_map: dict[str, int],
        chain_map: dict[str, int],
        accepted_candidates: set[Candidate],
        accepted_cluster_pairs: set[int],
):
    groups = {}
    for candidate in candidates:
        for k in candidate.subchain_pairs:
            k = frozenset(k); assert len(k) == 2
            groups.setdefault(k, []).append(candidate)

    def get_priority(pair):
        a, b = sorted(pair)

        prevalence = -len(groups[pair])
        same_chain = (0 if chain_map[a] == chain_map[b] else 1)
        tie_breaker = (a, b)  # to make the sort deterministic

        return prevalence, same_chain, tie_breaker

    for subchain_pair in sorted(groups, key=get_priority):
        cluster_pair = frozenset(
                cluster_map[k]
                for k in subchain_pair
        )
        if cluster_pair not in accepted_cluster_pairs:
            accepted_candidates.update(groups[subchain_pair])
            accepted_cluster_pairs.add(cluster_pair)

