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
import operator as op

from .database_io import open_db, transaction
from .util import tquiet
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from more_itertools import one, flatten
from functools import reduce, cached_property
from tqdm import tqdm

from typing import TypeAlias, TypeVar, Callable
from collections.abc import Iterable

Subchain: TypeAlias = tuple[str, int]
K, C = TypeVar('K'), TypeVar('C')

class Visitor:
    """
    The default implementation doesn't actually do any filtering.
    """

    def __init__(self, structure):
        raise NotImplementedError

    def propose(self, assembly):
        raise NotImplementedError

    def accept(self, candidates, memento):
        raise NotImplementedError

class Structure:

    def __init__(self, db, struct_id):
        self._db = db
        self._struct_id = struct_id

    def __repr__(self):
        return f'<Structure {self.pdb_id}>'

    @property
    def pdb_id(self):
        df = self._db.sql(
                'SELECT pdb_id FROM structure WHERE id = ?',
                params=[self._struct_id],
        ).pl()
        return df['pdb_id'].item()

    @cached_property
    def model_pdb_ids(self):
        df = self._db.sql(
                'SELECT pdb_id FROM model WHERE struct_id = ?',
                params=[self._struct_id],
        ).pl()
        return df['pdb_id'].to_list()

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
        return f'<Assembly {self.pdb_id}>'

    @cached_property
    def pdb_id(self):
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
    subchains: Iterable[Subchain] = frozenset()
    subchain_pairs: Iterable[tuple[Subchain, Subchain]] = frozenset()
    score: float = -1

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

        def __init__(self, _):
            pass

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
            # record the picked assemblies to the database.  This need doesn't 
            # violate any conventions, because this function isn't part of "the 
            # outside world".  However, because the Assembly class can't make 
            # this information public, it means that we need the concept of 
            # "module private" information.

            # Also note that we're prioritizing the subchains/subchain pairs 
            # based on the chain they appear in.  The goal is to favor 
            # subchains/subchain pairs that actually interact with each other.  
            # This method doesn't know what actual interactions are happening 
            # in the structure (custom visitors can be used for that), so we 
            # have to do the best with the information we have.  And one simple 
            # inference we can make is that subchains in the same chain are 
            # more likely to interact.


            chain_map = dict(
                    assembly._subchain_clusters
                    .select('subchain_id', 'chain_id')
                    .iter_rows()
            )
            subchain_ids = sorted(chain_map)

            for subchain_id in subchain_ids:
                yield Candidate(
                        subchains=[(subchain_id, 0)],
                        score=chain_map[subchain_id],
                )

            for subchain_pair in combinations(subchain_ids, r=2):
                subchain_1, subchain_2 = subchain_pair
                chain_1, chain_2 = chain_map[subchain_1], chain_map[subchain_2]

                prefer_same_chain = (0 if chain_1 == chain_2 else 1)
                prefer_early_chain = sorted((chain_1, chain_2))

                yield Candidate(
                        subchain_pairs=[((subchain_1, 0), (subchain_2, 0))],
                        score=(prefer_same_chain, *prefer_early_chain),
                )

        def accept(self, candidates, _):
            nonredundant.extend(
                    sorted(flatten(
                        [s for s, _ in c.subchains]
                        for c in candidates
                    ))
            )
            nonredundant_pairs.extend(
                    sorted(flatten(
                        [(s1, s2) for (s1,_), (s2,_) in c.subchain_pairs]
                        for c in candidates
                    ))
            )

    @dataclass(kw_only=True)
    class PickCandidate(Candidate):
        subchain_ids: list[int]

    visit_assemblies(db, PickVisitor, progress_factory=progress_factory)

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

def visit_assemblies(db, visitor_factory, *, memento=None, progress_factory=tquiet):
    """
    KBK: Below is an outline of the original algorithm I planned.  The final 
    version ended up a little different, but I haven't updated the notes yet.  

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
    ranked_subchains = db.sql('''\
            SELECT
                structure.id AS struct_id,
                structure.pdb_id AS struct_pdb_id,
                structure.rank AS struct_rank,
                assembly.id AS assembly_id,
                relevant_assemblies.rank AS assembly_rank,
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
            ORDER BY struct_rank, assembly_rank, chain_id, subchain_id
    ''').pl()

    # This isn't guaranteed to free the memory used by these data frames, but 
    # it at least makes it possible for that to happen.
    del relevant_subchains
    del relevant_assemblies

    if (last_assembly_id := memento._assembly_id) is not None:
        last_struct_rank, last_assembly_rank = \
                _select_assembly_rank(db, last_assembly_id)

        ranked_subchains = (
                ranked_subchains
                .filter(
                    (pl.col('struct_rank') > last_struct_rank) | (
                        (pl.col('struct_rank') == last_struct_rank) &
                        (pl.col('assembly_rank') > last_assembly_rank)
                    )
                )
        )

    n = ranked_subchains.n_unique('struct_id')
    progress = progress_factory(total=n)

    def all_clusters_redundant(subchain_clusters):
        clusters = set(subchain_clusters['cluster_id'])
        if clusters - memento._accepted_clusters:
            return False

        cluster_pairs = set(
                combinations_with_replacement(sorted(clusters), r=2)
        )
        if cluster_pairs - memento._accepted_cluster_pairs:
            return False

        return True

    for (struct_id, struct_pdb_id), struct_subchains_i in (
            ranked_subchains.group_by(
                ['struct_id', 'struct_pdb_id'],
                maintain_order=True,
            )
    ):
        progress.set_description(struct_pdb_id)
        progress.update()

        if all_clusters_redundant(struct_subchains_i):
            continue

        struct = Structure(db, struct_id)
        visitor = visitor_factory(struct)
        subchain_col = getattr(visitor, '_subchain_col', 'subchain_pdb_id')

        for (assembly_id,), assembly_subchains_j in (
                struct_subchains_i.group_by(
                    ['assembly_id'],
                    maintain_order=True,
                )
        ):
            if all_clusters_redundant(assembly_subchains_j):
                continue

            assembly = Assembly(db, assembly_id, assembly_subchains_j)
            candidates = list(visitor.propose(assembly))

            cluster_map = dict(
                    assembly_subchains_j
                    .select(subchain_col, 'cluster_id')
                    .iter_rows()
            )
            accepted_candidate_indices = set()

            _accept_nonredundant_subchains(
                    candidates,
                    cluster_map,
                    accepted_candidate_indices,
                    memento._accepted_clusters,
            )
            _accept_nonredundant_subchain_pairs(
                    candidates,
                    cluster_map,
                    accepted_candidate_indices,
                    memento._accepted_cluster_pairs,
            )

            accepted_candidates = [
                    candidates[i]
                    for i in sorted(accepted_candidate_indices)
            ]

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

    - It has been assigned a rank.  This indicates that (by some measures) it's 
      biologically relevant and not redundant.
    - It does not appear in a blacklisted structure.
    - It does not share a subchain cluster with any blacklisted assemblies.
    - It has a resolution below 10Å.  Assemblies that don't have a resolution 
      at all (e.g. from NMR structures) are not excluded by this criterion.  If 
      the assembly has multiple resolutions, only the best is considered.

    Not all of the assemblies returned by this function will necessarily end up 
    in the final dataset.  There are more redundancy checks to follow.  
    However, any assemblies not returned by this function will not be included 
    in the final dataset.
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
            SELECT assembly_id, rank
            FROM assembly_rank
            ANTI JOIN assembly_blacklist USING (assembly_id)
            ANTI JOIN assembly_low_res USING (assembly_id)
    ''').pl()

def _select_assembly_rank(db, assembly_id):
    df = db.sql('''\
            SELECT
                structure.rank AS struct_rank,
                assembly_rank.rank AS assembly_rank,
            FROM assembly
            JOIN structure ON structure.id = assembly.struct_id
            JOIN assembly_rank ON assembly.id = assembly_rank.assembly_id
            WHERE assembly.id = ?
    ''', params=[assembly_id]).pl()
    return one(df.iter_rows())

def _accept_nonredundant_subchains(
        candidates: list[Candidate],
        cluster_map: dict[str, int],
        accepted_candidate_indices: set[int],
        accepted_clusters: set[int],
):
    def iter_keys(candidate):
        yield from candidate.subchains

    def cluster_from_key(k):
        subchain, _ = k
        return cluster_map[subchain]

    _accept_nonredundant_candidates(
            candidates,
            iter_keys,
            cluster_from_key,
            accepted_candidate_indices,
            accepted_clusters,
    )

def _accept_nonredundant_subchain_pairs(
        candidates: list[Candidate],
        cluster_map: dict[str, int],
        accepted_candidate_indices: set[int],
        accepted_cluster_pairs: set[tuple[int, int]],
):
    def iter_keys(candidate):
        for pair in candidate.subchain_pairs:
            assert len(pair) == 2
            yield tuple(sorted(pair))

    def cluster_from_key(k):
        return tuple(sorted(
                cluster_map[subchain]
                for subchain, _ in k
        ))

    _accept_nonredundant_candidates(
            candidates,
            iter_keys,
            cluster_from_key,
            accepted_candidate_indices,
            accepted_cluster_pairs,
    )

def _accept_nonredundant_candidates(
        candidates: list[Candidate],
        iter_keys: Callable[[Candidate], Iterable[K]],
        cluster_from_key: Callable[[K], C],
        accepted_candidate_indices: set[int],
        accepted_clusters: set[C],
):
    groups = {}

    for i, candidate in enumerate(candidates):
        for k in iter_keys(candidate):
            group = groups.setdefault(k, dict(indices=[], scores=[]))
            group['indices'].append(i)
            group['scores'].append(candidate.score)

    def get_score(k):
        # Use `reduce()` instead of `sum()` because it allows the priorities to 
        # be any type, if the visitor can be sure that there will only be one 
        # candidate for each subchain/subchain pair.
        score = reduce(op.add, groups[k]['scores'])

        # This is just to guarantee that the sort is deterministic.
        tie_breaker = k

        return score, tie_breaker

    for k in sorted(groups, key=get_score):
        cluster = cluster_from_key(k)
        if cluster not in accepted_clusters:
            accepted_candidate_indices.update(groups[k]['indices'])
            accepted_clusters.add(cluster)
