import polars as pl
import macromol_census as mmc
import macromol_census.pick_assemblies as _mmc

from pytest_unordered import unordered

def test_pick_assemblies():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # The first assembly will always be added to the dataset.
    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='AXXX...'),
                dict(entity_id='2', type='polypeptide(L)', sequence='CXXX...'),
            ]),
    )

    # The second assembly will be in the same cluster as the first, so it will 
    # be excluded.
    mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='AXXX...'),
            ]),
    )

    # The third assembly has one subchain that the first doesn't, so it will be 
    # included for that subchain specifically.
    mmc.insert_structure(
            db, '3abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='AXXX...'),
                dict(entity_id='2', type='polypeptide(L)', sequence='DXXX...'),
            ]),
    )

    # The fourth assembly doesn't have any subchains that aren't already in 
    # earlier assemblies, but it does have a pair of subchains that hasn't 
    # occurred previously.
    mmc.insert_structure(
            db, '4abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='CXXX...'),
                dict(entity_id='2', type='polypeptide(L)', sequence='DXXX...'),
            ]),
    )

    # The fifth assembly has two copies of a subchain that has appeared 
    # previously.  This still counts as a unique pair.
    mmc.insert_structure(
            db, '5abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='AXXX...'),
                dict(entity_id='2', type='polypeptide(L)', sequence='AXXX...'),
            ]),
    )

    mmc.update_structure_ranks(
            db,
            pl.DataFrame([
                dict(struct_id=1, rank=1),
                dict(struct_id=2, rank=2),
                dict(struct_id=3, rank=3),
                dict(struct_id=4, rank=4),
                dict(struct_id=5, rank=5),
            ]),
    )

    mmc.insert_assembly_subchain_cover(
            db,
            pl.DataFrame([
                dict(assembly_id=1),
                dict(assembly_id=2),
                dict(assembly_id=3),
                dict(assembly_id=4),
                dict(assembly_id=5),
            ]),
    )

    mmc.insert_entity_clusters(
            db,
            pl.DataFrame([
                # AXXX...
                dict(cluster_id=1, entity_id=1),
                dict(cluster_id=1, entity_id=3),
                dict(cluster_id=1, entity_id=4),
                dict(cluster_id=1, entity_id=8),
                dict(cluster_id=1, entity_id=9),

                # CXXX...
                dict(cluster_id=2, entity_id=2),
                dict(cluster_id=2, entity_id=6),

                # DXXX...
                dict(cluster_id=3, entity_id=5),
                dict(cluster_id=3, entity_id=7),
            ]),
            'test',
    )

    _mmc.pick_assemblies(db)

    assert mmc.select_nonredundant_subchains(db).to_dicts() == [
            dict(subchain_id=1),
            dict(subchain_id=2),
            dict(subchain_id=5),
    ]
    assert mmc.select_nonredundant_subchain_pairs(db).to_dicts() == [
            dict(subchain_id_1=1, subchain_id_2=2),
            dict(subchain_id_1=4, subchain_id_2=5),
            dict(subchain_id_1=6, subchain_id_2=7),
            dict(subchain_id_1=8, subchain_id_2=9),
    ]

def test_pick_assemblies_chain():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # This structure has two chains, both of which contain two subchains of 
    # matching entities.  We want to make sure that the subchains that get 
    # picked are from the same chain, because otherwise they are unlikely to 
    # interact.  After that, we want to pick from alphabetically-nearby chains, 
    # for the same reason.
    #
    # Note that this isn't a perfect solution.  For example, you can still 
    # imagine picking non-adjacent monomers from an multimeric structure.  A 
    # proper solution would require working out the distances between each 
    # subchain when ingesting the mmCIF files (and all the coordinates are 
    # available).  But I think this would be expensive, and the benefit 
    # marginal, so for now I'd just using the chain heuristic.

    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
                dict(assembly_id='1', subchain_id='C'),
                dict(assembly_id='1', subchain_id='D'),
            ]),
            subchains=pl.DataFrame([
                # Deliberately put the subchains in such an order that we'd mix 
                # chains, if we didn't go out of our way not to.
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='1'),
                dict(id='C', chain_id='B', entity_id='2'),
                dict(id='D', chain_id='A', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='non-polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='XXXX...'),
            ]),
            monomer_entities=pl.DataFrame([
                dict(entity_id='2', comp_id='ABC'),
            ]),
    )

    mmc.insert_assembly_subchain_cover(
            db,
            pl.DataFrame([
                dict(assembly_id=1),
            ]),
    )

    _mmc.pick_assemblies(db)

    assert mmc.select_nonredundant_subchains(db).to_dicts() == [
            dict(subchain_id=1),
            dict(subchain_id=4),
    ]
    assert mmc.select_nonredundant_subchain_pairs(db).to_dicts() == unordered([
            dict(subchain_id_1=1, subchain_id_2=2),
            dict(subchain_id_1=1, subchain_id_2=4),
            dict(subchain_id_1=3, subchain_id_2=4),
    ])


def test_select_relevant_subchains():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # Create a structure with the following:
    # - ignored entity
    # - entities that cluster together
    # - entities that aren't in any cluster
    # - same entity appears in multiple subchains

    mmc.insert_structure(
            db, '1xyz',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
                dict(assembly_id='1', subchain_id='C'),
                dict(assembly_id='1', subchain_id='D'),
                dict(assembly_id='1', subchain_id='E'),
                dict(assembly_id='1', subchain_id='F'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
                dict(id='C', chain_id='C', entity_id='3'),
                dict(id='D', chain_id='A', entity_id='4'),
                dict(id='E', chain_id='B', entity_id='5'),
                dict(id='F', chain_id='C', entity_id='5'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
                dict(id='3', type='polymer', formula_weight_Da=None),
                dict(id='4', type='non-polymer', formula_weight_Da=None),
                dict(id='5', type='non-polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
                dict(entity_id='2', type='polypeptide(L)', sequence=None),
                dict(entity_id='3', type='polypeptide(L)', sequence=None),
            ]),
            monomer_entities=pl.DataFrame([
                dict(entity_id='4', comp_id='ABC'),
                dict(entity_id='5', comp_id='DEF'),
            ]),
    )

    clusters = pl.DataFrame([
        dict(entity_id=1, cluster_id=1),
        dict(entity_id=2, cluster_id=1),
    ])
    nonspecific_ligands = pl.DataFrame({'pdb_comp_id': ['ABC']})

    mmc.insert_entity_clusters(db, clusters, 'test')
    mmc.insert_nonspecific_ligands(db, nonspecific_ligands)

    assert _mmc._select_relevant_subchains(db).to_dicts() == [
            dict(subchain_id=1, cluster_id=1),
            dict(subchain_id=2, cluster_id=1),
            dict(subchain_id=3, cluster_id=2),
            dict(subchain_id=5, cluster_id=4),
            dict(subchain_id=6, cluster_id=4),
    ]
    
def test_select_relevant_assemblies():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # This structure has redundant assemblies (1 has the same subchains as 2 
    # and 3 together).  Only the first will be considered "relevant".

    struct_id_1 = mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
                dict(assembly_id='2', subchain_id='A'),
                dict(assembly_id='3', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='AXXX...'),
            ]),
    )

    # These two structure are both irrelevant.  The first is blacklisted, and 
    # the second has a subchain in the same cluster as a subchain from the 
    # blacklisted structure.
    
    struct_id_2 = mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='CXXX...'),
                dict(entity_id='2', type='polypeptide(L)', sequence='DXXX...'),
            ]),
    )

    struct_id_3 = mmc.insert_structure(
            db, '3abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='CXXX...'),
                dict(entity_id='2', type='polypeptide(L)', sequence='EXXX...'),
            ]),
    )

    mmc.insert_assembly_subchain_cover(
            db,
            pl.DataFrame([
                dict(assembly_id=1),
                dict(assembly_id=4),
                dict(assembly_id=5),
            ]),
    )

    mmc.insert_blacklisted_structures(
            db,
            pl.DataFrame([
                dict(pdb_id='2abc'),
            ]),
    )

    mmc.insert_entity_clusters(
            db,
            pl.DataFrame([
                dict(cluster_id=1, entity_id=2),
                dict(cluster_id=1, entity_id=4),
            ]),
            'test',
    )

    subchains = _mmc._select_relevant_subchains(db)
    assemblies = _mmc._select_relevant_assemblies(db, subchains)

    print(assemblies)

    assert assemblies.to_dicts() == [
            dict(assembly_id=1),
    ]
