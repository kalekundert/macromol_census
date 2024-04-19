import polars as pl
import macromol_census as mmc
import macromol_census.pick_assemblies
import sys

from pytest_unordered import unordered

# With just `import macromol_census.pick_assemblies as _mmc`, the function 
# `pick_assemblies()` ends up shadowing the module of the same name.
_mmc = sys.modules['macromol_census.pick_assemblies']

def test_pick_assemblies():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # The first assembly will always be added to the dataset.
    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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

    # The fifth assembly has two copies of the same subchain, which has 
    # appeared previously.  This still counts as a unique pair.
    mmc.insert_structure(
            db, '5abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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

    mmc.pick_assemblies(db)

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
    # marginal, so for now I'm just using the chain heuristic.

    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
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

    mmc.pick_assemblies(db)

    assert mmc.select_nonredundant_subchains(db).to_dicts() == [
            dict(subchain_id=1),
            dict(subchain_id=4),
    ]
    assert mmc.select_nonredundant_subchain_pairs(db).to_dicts() == unordered([
            dict(subchain_id_1=1, subchain_id_2=2),
            dict(subchain_id_1=1, subchain_id_2=4),
            dict(subchain_id_1=3, subchain_id_2=4),
    ])

def test_visit_assemblies_memento(tmp_path):
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # `1abc` and `3abc` both have just one assembly, while `2abc` has two.  The 
    # idea is to stop in the middle of `2abc` (i.e. after its first assembly 
    # and before its second) and then restart from that point.

    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            models=pl.DataFrame([
                dict(id='1'),
            ]),
            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='A', entity_id='2'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
                dict(id='2', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
                dict(entity_id='2', type='polypeptide(L)', sequence=None),
            ]),
    )
    mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            models=pl.DataFrame([
                dict(id='1'),
            ]),
            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
                dict(id='2', type=None, polymer_count=1),
            ]),
            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='2', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
            ]),
    )
    mmc.insert_structure(
            db, '3abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            models=pl.DataFrame([
                dict(id='1'),
                dict(id='2'),
            ]),
            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
            ]),
    )
    mmc.update_structure_ranks(
            db,
            pl.DataFrame([
                dict(struct_id=1, rank=1),
                dict(struct_id=2, rank=2),
                dict(struct_id=3, rank=3),
            ]),
    )
    mmc.insert_assembly_subchain_cover(
            db,
            pl.DataFrame([
                dict(assembly_id=1),
                dict(assembly_id=2),
                dict(assembly_id=3),
                dict(assembly_id=4),
            ]),
    )
    mmc.insert_entity_clusters(
            db,
            pl.DataFrame([
                dict(cluster_id=1, entity_id=1),
                dict(cluster_id=2, entity_id=2),
                dict(cluster_id=3, entity_id=3),
                dict(cluster_id=4, entity_id=4),
            ]),
            'test',
    )

    mmc.pick_assemblies(db)
    iterations = 0
    visited_assemblies = []
    memento_path = tmp_path / 'memento.pkl'

    class MockVisitor(mmc.Visitor):

        def __init__(self, structure):
            assert '{' not in repr(structure)
            self.structure = structure

        def propose(self, assembly):
            assert '{' not in repr(assembly)

            # Record every PDB id, just to make sure they're all calculated 
            # correctly.
            tag = (
                    self.structure.pdb_id,
                    self.structure.model_pdb_ids,
                    assembly.pdb_id,
                    assembly.subchain_pdb_ids,
            )
            visited_assemblies.append(tag)
            yield from []

        def accept(self, candidates, memento):
            memento.save(memento_path)

            nonlocal iterations
            iterations += 1
            if iterations == 2:
                raise MockInterruption

    class MockInterruption(Exception):
        pass

    try:
        mmc.visit_assemblies(db, MockVisitor)
    except MockInterruption:
        pass

    assert visited_assemblies == [
            ('1abc', ['1'], '1', ['A', 'B']),
            ('2abc', ['1'], '1', ['A']),
    ]

    memento = mmc.Memento.load(memento_path)
    mmc.visit_assemblies(db, MockVisitor, memento=memento)

    assert visited_assemblies == [
            ('1abc', ['1'], '1', ['A', 'B']),
            ('2abc', ['1'], '1', ['A']),
            ('2abc', ['1'], '2', ['B']),
            ('3abc', ['1', '2'], '1', ['A']),
    ]

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

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=3),
            ]),
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

    # Subchain cover
    # ==============
    # `1abc` has redundant assemblies (1 has the same subchains as 2 and 3 
    # together).  Only the first will be part of the "subchain cover", 
    # therefore only the first will be considered "relevant".

    struct_id_1 = mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
                dict(id='2', type=None, polymer_count=1),
                dict(id='3', type=None, polymer_count=1),
            ]),
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

    # Structure blacklist
    # ===================
    # `2abc` is irrelevant because it will be placed on the "blacklist":
    
    struct_id_2 = mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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

    # `3abc` is irrelevant because it has a subchain that belongs to the same 
    # cluster as a subchain from `2abc`, which is blacklisted.
    struct_id_3 = mmc.insert_structure(
            db, '3abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=2),
            ]),
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

    assert assemblies.to_dicts() == [
            dict(assembly_id=1),
    ]

def test_select_relevant_assemblies_resolution():
    # I decided to separate out the tests for the low resolution structures 
    # from all the other assembly-filtering criteria.  There are a lot of cases 
    # to consider here, and I didn't want to make the other test too confusing.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # `1abc` is relevant because its crystallography resolution is high enough.
    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence='XXX...'),
            ]),
            xtal_quality=pl.DataFrame([
                dict(resolution_A=9.9, r_work=None, r_free=None),
            ]),
    )

    # `2abc` is irrelevant because its crystallography resolution is too low. 
    mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence='XXX...'),
            ]),
            xtal_quality=pl.DataFrame([
                dict(resolution_A=10, r_work=None, r_free=None),
            ]),
    )

    # `3abc` is irrelevant because its EM resolution is too low. 
    mmc.insert_structure(
            db, '3abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence='FXXX...'),
            ]),
            em_quality=pl.DataFrame([
                dict(resolution_A=10),
            ]),
    )

    # `4abc` is relevant because one of its crystallography resolutions is high 
    # enough, even though others are too low.
    mmc.insert_structure(
            db, '4abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence='FXXX...'),
            ]),
            xtal_quality=pl.DataFrame([
                dict(resolution_A=10, r_work=None, r_free=None),
                dict(resolution_A=4, r_work=None, r_free=None),
            ]),
            em_quality=pl.DataFrame([
                dict(resolution_A=10),
            ]),
    )

    # `5abc` is relevant because one of its EM resolutions is high enough, even 
    # though others are too low.
    mmc.insert_structure(
            db, '5abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence='FXXX...'),
            ]),
            xtal_quality=pl.DataFrame([
                dict(resolution_A=10, r_work=None, r_free=None),
            ]),
            em_quality=pl.DataFrame([
                dict(resolution_A=10),
                dict(resolution_A=4),
            ]),
    )

    # `6abc` is relevant because it doesn't have a resolution (e.g. it's an NMR 
    # structure).
    mmc.insert_structure(
            db, '6abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
            ]),
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
                dict(entity_id='1', type='polypeptide(L)', sequence='FXXX...'),
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
                dict(assembly_id=6),
            ]),
    )

    subchains = _mmc._select_relevant_subchains(db)
    assemblies = _mmc._select_relevant_assemblies(db, subchains)

    assert assemblies.to_dicts() == [
            dict(assembly_id=1),
            dict(assembly_id=4),
            dict(assembly_id=5),
            dict(assembly_id=6),
    ]

def test_select_assembly_rank():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # Both structures have two assemblies.  The first is lower resolution than 
    # the first, so should have a lower rank.

    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
                dict(id='2', type=None, polymer_count=1),
            ]),
            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='2', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
            ]),
            xtal_quality=pl.DataFrame({
                'resolution_A': [3.0],
                'r_work': [None],
                'r_free': [None],
            }),
    )
    mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=1),
                dict(id='2', type=None, polymer_count=1),
            ]),
            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='2', subchain_id='B'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
            ]),
            xtal_quality=pl.DataFrame({
                'resolution_A': [1.0],
                'r_work': [None],
                'r_free': [None],
            }),
    )

    ranks = mmc.rank_structures(db)
    mmc.update_structure_ranks(db, ranks)

    assert _mmc._select_assembly_rank(db, 1) == 2
    assert _mmc._select_assembly_rank(db, 2) == 2
    assert _mmc._select_assembly_rank(db, 3) == 1
    assert _mmc._select_assembly_rank(db, 4) == 1

def test_accept_nonredundant_subchains():
    # - Empty candidate is ignored, without causing any problems.
    #
    # - A is excluded because it's part of a cluster that's already been 
    #   accepted.
    #
    # - B and C are in the same cluster.  C gets accepted, because it appears 
    #   more frequently.  The candidate that contains both B and C also gets 
    #   accepted; just one unique feature is required to be accepted.
    #
    # - D and E are in the same cluster.  E gets accepted because it's from an 
    #   earlier chain.  The purpose of this ranking is to encourage picking 
    #   subchains from the same chain (assuming they'll be more likely to 
    #   interact) in the absence of any other distinguishing features.
    #
    # - F and G are in the same cluster and the same chain.  F gets picked 
    #   because it comes first alphabetically.  This is just a tie-breaker to 
    #   make the sorting algorithm deterministic.

    candidates = [
            mmc.Candidate(),
            mmc.Candidate(subchains=['A']),
            mmc.Candidate(subchains=['B']),
            mmc.Candidate(subchains=['B', 'C']),    # accepted
            mmc.Candidate(subchains=['C']),         # accepted
            mmc.Candidate(subchains=['C']),         # accepted
            mmc.Candidate(subchains=['D']),
            mmc.Candidate(subchains=['E']),         # accepted
            mmc.Candidate(subchains=['F']),         # accepted
            mmc.Candidate(subchains=['G']),
    ]
    cluster_map = {
            'A': 1,
            'B': 2, 'C': 2,
            'D': 3, 'E': 3,
            'F': 4, 'G': 4,
    }
    chain_map = {
            'A': 'A',
            'B': 'B',
            'C': 'C',
            'D': 'E',
            'E': 'D',
            'F': 'F',
            'G': 'F',
    }
    accepted_clusters = {1}
    accepted_candidate_indices = set()

    _mmc._accept_nonredundant_subchains(
            candidates,
            cluster_map,
            chain_map,
            accepted_candidate_indices,
            accepted_clusters,
    )

    assert accepted_candidate_indices == {3, 4, 5, 7, 8}
    assert accepted_clusters == {1,2,3,4}

def test_accept_nonredundant_subchain_pairs():
    # - Empty candidate is ignored, without causing any problems.
    #
    # - (A,B) is excluded because its cluster pair has already been accepted.
    #
    # - (C,D) and (C,E) have the same cluster pair.  (C,E) gets accepted, 
    #   because it appears more frequently.  The candidate that contains both 
    #   pairs also gets accepted.  It just matters that the candidate has a 
    #   unique pair, it doesn't matter if it also has non-unique pairs.
    #
    # - (F,G) and (F,H) have the same cluster pair.  (F,H) is accepted because 
    #   both subchains belong to the same chain, unlike (F,G).  In the absence 
    #   of any other information, this indicates that F and H are more likely 
    #   to be truly interacting.
    #
    # - (I,J) and (I,K) have the same cluster pair and belong to the same 
    #   chain.  (I,J) is accepted because it comes first alphabetically.  This 
    #   is just a tie-breaker to make the sorting algorithm deterministic.  
    #   This cluster pair also features the same cluster twice, rather than two 
    #   different clusters.  This shouldn't affect the logic at all.

    candidates = [
            mmc.Candidate(),

            mmc.Candidate(subchain_pairs=[('A','B')]),
            mmc.Candidate(subchain_pairs=[('B','A')]),

            mmc.Candidate(subchain_pairs=[('C','D')]),
            mmc.Candidate(subchain_pairs=[('D','C'), ('C','E')]),   # accepted
            mmc.Candidate(subchain_pairs=[('C','E')]),              # accepted
            mmc.Candidate(subchain_pairs=[('E','C')]),              # accepted

            mmc.Candidate(subchain_pairs=[('F','G')]),
            mmc.Candidate(subchain_pairs=[('F','H')]),              # accepted

            mmc.Candidate(subchain_pairs=[('J','I')]),              # accepted
            mmc.Candidate(subchain_pairs=[('I','K')]),
    ]
    cluster_map = {
            'A': 1,
            'B': 2,
            'C': 3,
            'D': 4,
            'E': 4,
            'F': 5,
            'G': 6,
            'H': 6,
            'I': 7,
            'J': 7,
            'K': 7,
    }
    chain_map = {
            'A': 'A',
            'B': 'A',
            'C': 'B',
            'D': 'B',
            'E': 'B',
            'F': 'C',
            'G': 'D',
            'H': 'C',
            'I': 'E',
            'J': 'E',
            'K': 'E',
    }
    accepted_cluster_pairs = {
            frozenset((1,2)),
    }
    accepted_candidate_indices = set()

    _mmc._accept_nonredundant_subchain_pairs(
            candidates,
            cluster_map,
            chain_map,
            accepted_candidate_indices,
            accepted_cluster_pairs,
    )

    assert accepted_candidate_indices == {4, 5, 6, 8, 9}
    assert accepted_cluster_pairs == {
            frozenset((1,2)),
            frozenset((3,4)),
            frozenset((5,6)),
            frozenset((7,7)),
    }
