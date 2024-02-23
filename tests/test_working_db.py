import macromol_census.working_db as mmc
import polars as pl
import pytest

from polars.testing import assert_frame_equal
from pytest import approx
from datetime import date
from functools import partial

assert_frame_equal = partial(assert_frame_equal, check_dtype=False)

@pytest.fixture
def db():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    assembly_chain_pairs = pl.DataFrame([
        dict(assembly_id='1', chain_id='A'),
        dict(assembly_id='1', chain_id='B'),
    ])
    chain_entity_pairs = pl.DataFrame([
        dict(chain_id='A', entity_id='1'),
        dict(chain_id='A', entity_id='2'),
        dict(chain_id='B', entity_id='1'),
        dict(chain_id='B', entity_id='2'),
    ])
    polymers = pl.DataFrame([
        dict(
            entity_id='1',
            type='polypeptide(L)',
            sequence='DDWEIPDGQI...',
        ),
    ])
    nonpolymers = pl.DataFrame([
        dict(entity_id='2', comp_id='1SU'),
    ])

    # I don't think there are any structures that combine crystallography, NMR, 
    # and EM data.  But it's not impossible, and it's convenient for testing to 
    # have reasonable values in every table.

    quality_xtal = pl.DataFrame([
        dict(resolution_A=1.5, num_reflections=41000, r_work=0.16, r_free=0.18),
    ])
    quality_nmr = pl.DataFrame([
        dict(conformer_id='1'),
    ])
    quality_em = pl.DataFrame([
        dict(resolution_A=3.4),
    ])

    mmc.insert_model(
            db, '9xyz',
            exptl_methods=['X-RAY DIFFRACTION'],
            deposit_date=date(year=2024, month=2, day=16),
            num_atoms=1000,
            assembly_chain_pairs=assembly_chain_pairs,
            chain_entity_pairs=chain_entity_pairs,
            polymers=polymers,
            nonpolymers=nonpolymers,
            quality_xtal=quality_xtal,
            quality_nmr=quality_nmr,
            quality_em=quality_em,
    )

    return db

def test_model(db):
    assert mmc.select_models(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='9xyz',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2024, month=2, day=16),
                num_atoms=1000,
            ),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(id=1, model_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, model_id=1, pdb_id='A'),
            dict(id=2, model_id=1, pdb_id='B'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(id=1, model_id=1, pdb_id='1'),
            dict(id=2, model_id=1, pdb_id='2'),
    ]
    assert mmc.select_assembly_chain_pairs(db).to_dicts() == [
            dict(assembly_id=1, chain_id=1),
            dict(assembly_id=1, chain_id=2),
    ]
    assert mmc.select_chain_entity_pairs(db).to_dicts() == [
            dict(chain_id=1, entity_id=1),
            dict(chain_id=1, entity_id=2),
            dict(chain_id=2, entity_id=1),
            dict(chain_id=2, entity_id=2),
    ]
    assert mmc.select_qualities_xtal(db).to_dicts() == [
            dict(model_id=1, resolution_A=approx(1.5), num_reflections=41000, r_work=approx(0.16), r_free=approx(0.18)),
    ]
    assert mmc.select_qualities_nmr(db).to_dicts() == [
            dict(model_id=1, pdb_conformer_id='1', num_dist_restraints=None),
    ]
    assert mmc.select_qualities_em(db).to_dicts() == [
            dict(model_id=1, resolution_A=approx(3.4), q_score=None),
    ]

def test_model_blacklist(db):
    blacklist = pl.DataFrame([
        dict(pdb_id='9xyz'),
    ])

    mmc.insert_model_blacklist(db, blacklist)

    assert mmc.select_model_blacklist(db).to_dicts() == [
        dict(model_id=1),
    ]

def test_quality_nmr(db):
    mmc.update_quality_nmr(db, 1, num_dist_restraints=162)

    assert mmc.select_qualities_nmr(db).to_dicts() == [
            dict(model_id=1, pdb_conformer_id='1', num_dist_restraints=162),
    ]

def test_quality_em(db):
    mmc.insert_quality_em(db, 1, resolution_A=3.3, q_score=0.5)
    
    assert mmc.select_qualities_em(db).to_dicts() == [
            dict(model_id=1, resolution_A=approx(3.4), q_score=None),
            dict(model_id=1, resolution_A=approx(3.3), q_score=approx(0.5)),
    ]

def test_quality_clashscore(db):
    mmc.insert_quality_clashscore(db, 1, clashscore=27.2)

    assert mmc.select_qualities_clashscore(db).to_dicts() == [
            dict(model_id=1, clashscore=approx(27.2)),
    ]

def test_chain_clusters(db):
    chain_clusters = pl.DataFrame([
        dict(cluster=1, chain_id=1),
        dict(cluster=1, chain_id=2),
    ])

    mmc.insert_chain_clusters(db, chain_clusters)

    assert mmc.select_chain_clusters(db).to_dicts() == [
        dict(cluster=1, chain_id=1),
        dict(cluster=1, chain_id=2),
    ]

def test_entity_clusters(db):
    entity_clusters = pl.DataFrame([
        dict(cluster=1, entity_id=1),
        dict(cluster=1, entity_id=2),
    ])

    mmc.insert_entity_clusters(db, entity_clusters)

    assert mmc.select_entity_clusters(db).to_dicts() == [
        dict(cluster=1, entity_id=1),
        dict(cluster=1, entity_id=2),
    ]
