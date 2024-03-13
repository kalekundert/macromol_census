import networkx as nx
import macromol_census as mmc

def make_branched_entity_graph(entity_id, *edges):
    g = nx.Graph(entity_id=entity_id)

    for seq_1, comp_1, atom_1, seq_2, comp_2, atom_2, bond in edges:
        atom_1 = comp_1, atom_1
        atom_2 = comp_2, atom_2

        g.add_node(seq_1, label=comp_1)
        g.add_node(seq_2, label=comp_2)
        g.add_node(atom_1, label=atom_1)
        g.add_node(atom_2, label=atom_2)

        g.add_edge(seq_1, atom_1, label=None)
        g.add_edge(atom_1, atom_2, label=bond)
        g.add_edge(atom_2, seq_2, label=None)

    return g

def test_cluster_isomorphic_graphs():
    entities = [
            make_branched_entity_graph(
                1,
                (1, 'ABC', 'C1', 2, 'XYZ', 'O3', 'sing'),
            ),

            # Same as 1, but with the nodes in a different order.
            make_branched_entity_graph(
                2,
                (2, 'XYZ', 'O3', 1, 'ABC', 'C1', 'sing'),
            ),

            # Different residue name than 1.
            make_branched_entity_graph(
                3,
                (1, '___', 'C1', 2, 'XYZ', 'O3', 'sing'),
            ),

            # Different atom name than 1.
            make_branched_entity_graph(
                4,
                (1, 'ABC', '__', 2, 'XYZ', 'O3', 'sing'),
            ),

            # Different bond order name than 1.
            make_branched_entity_graph(
                5,
                (1, 'ABC', 'C1', 2, 'XYZ', 'O3', 'doub'),
            ),
    ]

    assert mmc.cluster_isomorphic_graphs(entities).to_dicts() == [
            dict(entity_id=1, cluster_id=1),
            dict(entity_id=2, cluster_id=1),
            dict(entity_id=3, cluster_id=2),
            dict(entity_id=4, cluster_id=3),
            dict(entity_id=5, cluster_id=4),
    ]
