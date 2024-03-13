"""
Cluster branched entities by identity.

Usage:
    mmc_find_identical_branched_entities <in:db>

The PDB defines three kinds of molecular entities: polymer, non-polymer, and 
branched [1].  Branched entities, those where each monomer can be bonded to 
more than one other monomer, are the most complicated.  Most, possibly all, of 
these entities are oligosaccharides.

When deciding which assemblies to include in the final non-redundant dataset, 
it's important to decide which entities to consider equivalent.  For polymers, 
we do that using sequence alignments.  For non-polymers, we do that using the 
PDB "component id".  Branch entities are not so easy to compare, because there 
are many ways to specify the same set of bonds.  In fact, making such 
comparisons is an example of the graph isomorphism problem.

This script identifies every group of identical branched entities in the PDB.  
It does so by (i) creating a graph representing each branched entity, (ii) 
hashing the graph to find groups of potentially identical entities, and (iii) 
solving the graph isomorphism problem within each group.

Note that this script does not make any attempt to group "similar" entities, 
like a sequence alignment does.  This is mostly because there aren't enough 
branched entities in the PDB to judge how similar two entities need to be to be 
grouped together.

[1] There are actually a few other kinds as well, but these three are the ones 
    relevant for this discussion.
"""

import polars as pl
import networkx as nx
import operator as op

from .database_io import (
        open_db, select_branched_entity_bonds, insert_entity_clusters,
)
from collections import defaultdict
from tqdm import tqdm

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])

    clusters = find_identical_branched_entities(db)
    insert_entity_clusters(db, clusters, 'identical-branched-entities')

def find_identical_branched_entities(db):
    entities = []
    bonds = select_branched_entity_bonds(db)
    n = db.sql('''\
            SELECT count(*)
            FROM entity
            WHERE type = 'branched'
    ''').pl().item()

    for (entity_id,), bonds_i in tqdm(
            bonds.group_by(['entity_id']),
            desc='finding branched entities',
            total=n,
    ):
        g = nx.Graph(entity_id=entity_id)

        for row in bonds_i.iter_rows(named=True):
            comp_1 = row['pdb_seq_id_1']
            atom_1 = (comp_1, row['pdb_atom_id_1'])
            comp_2 = row['pdb_seq_id_2']
            atom_2 = (comp_2, row['pdb_atom_id_2'])

            g.add_node(comp_1, label=row['pdb_comp_id_1'])
            g.add_node(comp_2, label=row['pdb_comp_id_2'])
            g.add_node(atom_1, label=atom_1)
            g.add_node(atom_2, label=atom_1)

            g.add_edge(comp_1, atom_1, label=None)
            g.add_edge(atom_1, atom_2, label=row['bond_order'])
            g.add_edge(atom_2, comp_2, label=None)

        entities.append(g)

    return cluster_isomorphic_graphs(entities)

def cluster_isomorphic_graphs(graphs):
    candidate_groups = defaultdict(list)

    for g in tqdm(graphs, desc='computing graph hashes'):
        hash_ = nx.weisfeiler_lehman_graph_hash(
                g,
                node_attr='label',
                edge_attr='label',
        )
        candidate_groups[hash_].append(g)

    rows = []
    cursor = 1

    for candidate_group in tqdm(
            candidate_groups.values(),
            desc='computing graph isomorphisms',
    ):
        confirmed_groups = []

        for g in candidate_group:
            for subgroup in confirmed_groups:
                if nx.is_isomorphic(g, subgroup[0], op.eq, op.eq):
                    subgroup.append(g)
                    break
            else:
                confirmed_groups.append([g])

        for confirmed_group in confirmed_groups:
            for g in confirmed_group:
                rows.append({
                    **g.graph,
                    'cluster_id': cursor,
                })
            cursor += 1

    return pl.DataFrame(rows)
