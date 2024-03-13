import macromol_census as mmc
import polars as pl
import pytest

from polars.testing import assert_frame_equal
from pytest import approx
from datetime import date
from functools import partial

assert_frame_equal = partial(assert_frame_equal, check_dtype=False)

def insert_1abc(db):
    # A structure with the minimal amount of information.
    mmc.insert_structure(
            db, '1abc',
            exptl_methods=['X-RAY DIFFRACTION'],
            deposit_date=date(year=2024, month=2, day=16),
            full_atom=True,

            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=18_000),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='MGPG...'),
            ]),
    )

def insert_9xyz(db):
    # A structure with as many different kinds of information as possible.

    # This fictional structure is a protein dimer.  One chain is bound to a 
    # monomer (i.e. a small molecule) and the other is bound to a branched 
    # oligomer (i.e. a sugar).  The asymmetric unit has two copies of this 
    # complex, both of which are separate biological assemblies.  The model 
    # somehow combines data from X-ray crystallography, EM, and NMR.

    mmc.insert_structure(
            db, '9xyz',
            exptl_methods=[
                'X-RAY DIFFRACTION',
                'ELECTRON MICROSCOPY',
                'SOLUTION NMR',
            ],
            deposit_date=date(year=2024, month=3, day=1),
            full_atom=True,

            models=pl.DataFrame([
                dict(id='1'),
                dict(id='2'),
            ]),
            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
                dict(assembly_id='1', subchain_id='C'),
                dict(assembly_id='1', subchain_id='D'),
                dict(assembly_id='2', subchain_id='E'),
                dict(assembly_id='2', subchain_id='F'),
                dict(assembly_id='2', subchain_id='G'),
                dict(assembly_id='2', subchain_id='H'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='A', entity_id='2'),
                dict(id='C', chain_id='B', entity_id='3'),
                dict(id='D', chain_id='B', entity_id='4'),
                dict(id='E', chain_id='C', entity_id='1'),
                dict(id='F', chain_id='C', entity_id='2'),
                dict(id='G', chain_id='D', entity_id='3'),
                dict(id='H', chain_id='D', entity_id='4'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=10_000),
                dict(id='2', type='non-polymer', formula_weight_Da=200),
                dict(id='3', type='polymer', formula_weight_Da=30_000),
                dict(id='4', type='branched', formula_weight_Da=400),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type='polypeptide(L)', sequence='MNTP...'),
                dict(entity_id='3', type='polypeptide(L)', sequence='DDWE...'),
            ]),
            monomer_entities=pl.DataFrame([
                dict(entity_id='2', comp_id='EQU'),
            ]),
            branched_entities=pl.DataFrame([
                dict(entity_id='4', type='oligosaccharide'),
            ]),
            branched_entity_bonds=pl.DataFrame([
                dict(
                    entity_id='4',
                    seq_id_1='1',
                    comp_id_1='NAG',
                    atom_id_1='C1',
                    seq_id_2='2',
                    comp_id_2='NAG',
                    atom_id_2='O4',
                    bond_order='sing',
                ),
            ]),
            xtal_quality=pl.DataFrame([
                dict(
                    resolution_A=1.5,
                    r_work=0.16,
                    r_free=0.18,
                ),
            ]),
            nmr_representative='2',
            em_quality=pl.DataFrame([
                dict(resolution_A=3.4),
            ]),
    )


def test_insert_structure():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='1abc',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2024, month=2, day=16),
                full_atom=True,
                rank=None,
            ),
            dict(
                id=2,
                pdb_id='9xyz',
                exptl_methods=[
                    'X-RAY DIFFRACTION',
                    'ELECTRON MICROSCOPY',
                    'SOLUTION NMR',
                ],
                deposit_date=date(year=2024, month=3, day=1),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            dict(id=1, struct_id=2, pdb_id='1'),
            dict(id=2, struct_id=2, pdb_id='2'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
            dict(id=2, struct_id=2, pdb_id='A'),
            dict(id=3, struct_id=2, pdb_id='B'),
            dict(id=4, struct_id=2, pdb_id='C'),
            dict(id=5, struct_id=2, pdb_id='D'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1', type='polymer', formula_weight_Da=approx(18_000)),
            dict(id=2, struct_id=2, pdb_id='1', type='polymer', formula_weight_Da=approx(10_000)),
            dict(id=3, struct_id=2, pdb_id='2', type='non-polymer', formula_weight_Da=approx(200)),
            dict(id=4, struct_id=2, pdb_id='3', type='polymer', formula_weight_Da=approx(30_000)),
            dict(id=5, struct_id=2, pdb_id='4', type='branched', formula_weight_Da=approx(400)),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(entity_id=1, type='polypeptide(L)', sequence='MGPG...'),
            dict(entity_id=2, type='polypeptide(L)', sequence='MNTP...'),
            dict(entity_id=4, type='polypeptide(L)', sequence='DDWE...'),
    ]
    assert mmc.select_branched_entities(db).to_dicts() == [
            dict(entity_id=5, type='oligosaccharide'),
    ]
    assert mmc.select_branched_entity_bonds(db).to_dicts() == [
            dict(
                entity_id=5,
                pdb_seq_id_1='1', pdb_comp_id_1='NAG', pdb_atom_id_1='C1',
                pdb_seq_id_2='2', pdb_comp_id_2='NAG', pdb_atom_id_2='O4',
                bond_order='sing',
            ),
    ]
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=3, pdb_comp_id='EQU'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1, chain_id=1, entity_id=1, pdb_id='A'),
            dict(id=2, chain_id=2, entity_id=2, pdb_id='A'),
            dict(id=3, chain_id=2, entity_id=3, pdb_id='B'),
            dict(id=4, chain_id=3, entity_id=4, pdb_id='C'),
            dict(id=5, chain_id=3, entity_id=5, pdb_id='D'),
            dict(id=6, chain_id=4, entity_id=2, pdb_id='E'),
            dict(id=7, chain_id=4, entity_id=3, pdb_id='F'),
            dict(id=8, chain_id=5, entity_id=4, pdb_id='G'),
            dict(id=9, chain_id=5, entity_id=5, pdb_id='H'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1'),
            dict(id=2, struct_id=2, pdb_id='1'),
            dict(id=3, struct_id=2, pdb_id='2'),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=2, subchain_id=2),
            dict(assembly_id=2, subchain_id=3),
            dict(assembly_id=2, subchain_id=4),
            dict(assembly_id=2, subchain_id=5),
            dict(assembly_id=3, subchain_id=6),
            dict(assembly_id=3, subchain_id=7),
            dict(assembly_id=3, subchain_id=8),
            dict(assembly_id=3, subchain_id=9),
    ]
    assert mmc.select_xtal_quality(db).to_dicts() == [
            dict(
                struct_id=2,
                source='mmcif_pdbx',
                resolution_A=approx(1.5),
                r_work=approx(0.16),
                r_free=approx(0.18),
            ),
    ]
    assert mmc.select_nmr_representatives(db).to_dicts() == [
            dict(model_id=2, source='mmcif_pdbx'),
    ]
    assert mmc.select_em_quality(db).to_dicts() == [
            dict(
                struct_id=2,
                source='mmcif_pdbx',
                resolution_A=approx(3.4),
                q_score=None,
            ),
    ]

def test_update_structure_ranks():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    ranks = pl.DataFrame([
        dict(struct_id=1, rank=2),
        dict(struct_id=2, rank=1),
    ])
    mmc.update_structure_ranks(db, ranks)

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='1abc',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2024, month=2, day=16),
                full_atom=True,
                rank=2,
            ),
            dict(
                id=2,
                pdb_id='9xyz',
                exptl_methods=[
                    'X-RAY DIFFRACTION',
                    'ELECTRON MICROSCOPY',
                    'SOLUTION NMR',
                ],
                deposit_date=date(year=2024, month=3, day=1),
                full_atom=True,
                rank=1,
            ),
    ]

def test_insert_blacklisted_structures():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    blacklist = pl.DataFrame([
        dict(pdb_id='9xyz'),
    ])

    mmc.insert_blacklisted_structures(db, blacklist)

    assert mmc.select_blacklisted_structures(db).to_dicts() == [
        dict(struct_id=2),
    ]

def test_insert_nmr_quality():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    mmc.insert_nmr_quality(
            db, 2,
            source='mmcif_pdbx_vrpt',
            num_dist_restraints=162,
    )

    assert mmc.select_nmr_quality(db).to_dicts() == [
            dict(
                struct_id=2,
                source='mmcif_pdbx_vrpt',
                num_dist_restraints=162,
            ),
    ]

def test_insert_em_quality():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    mmc.insert_em_quality(
            db, 2,
            source='mmcif_pdbx_vrpt',
            resolution_A=3.3,
            q_score=0.5,
    )
    
    assert mmc.select_em_quality(db).to_dicts() == [
            dict(
                struct_id=2,
                source='mmcif_pdbx',
                resolution_A=approx(3.4),
                q_score=None,
            ),
            dict(
                struct_id=2,
                source='mmcif_pdbx_vrpt',
                resolution_A=approx(3.3),
                q_score=approx(0.5),
            ),
    ]

def test_insert_clashscore():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    mmc.insert_clashscore(
            db, 2,
            source='mmcif_pdbx_vrpt',
            clashscore=27.2,
    )

    assert mmc.select_clashscores(db).to_dicts() == [
            dict(
                struct_id=2,
                source='mmcif_pdbx_vrpt',
                clashscore=approx(27.2),
            ),
    ]

def test_insert_entity_clusters():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    entity_clusters = pl.DataFrame([
        dict(cluster_id='A', entity_id=1),
        dict(cluster_id='A', entity_id=2),
        dict(cluster_id='B', entity_id=3),
        dict(cluster_id='B', entity_id=4),
        dict(cluster_id='C', entity_id=5),
    ])

    mmc.insert_entity_clusters(db, entity_clusters, 'test')

    assert mmc.select_clusters(db).to_dicts() == [
        dict(id=1, namespace='test', name='A'),
        dict(id=2, namespace='test', name='B'),
    ]
    assert mmc.select_entity_clusters(db).to_dicts() == [
        dict(entity_id=1, cluster_id=1),
        dict(entity_id=2, cluster_id=1),
        dict(entity_id=3, cluster_id=2),
        dict(entity_id=4, cluster_id=2),
    ]
