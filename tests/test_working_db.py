import macromol_census.working_db as mmc
import polars as pl

from polars.testing import assert_frame_equal
from datetime import date
from functools import partial

assert_frame_equal = partial(assert_frame_equal, check_dtype=False)

def test_model():
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

    mmc.insert_model(
            db, '9xyz',
            exptl_methods=['X-RAY DIFFRACTION'],
            deposit_date=date(year=2024, month=2, day=16),
            num_atoms=1000,
            assembly_chain_pairs=assembly_chain_pairs,
            chain_entity_pairs=chain_entity_pairs,
            polymers=polymers,
            nonpolymers=nonpolymers,
    )

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

    chain_clusters = pl.DataFrame([
        dict(cluster=1, chain_id=1),
        dict(cluster=1, chain_id=2),
    ])
    entity_clusters = pl.DataFrame([
        dict(cluster=1, entity_id=1),
        dict(cluster=1, entity_id=2),
    ])

    mmc.insert_chain_clusters(db, chain_clusters)
    mmc.insert_entity_clusters(db, entity_clusters)

    assert mmc.select_chain_clusters(db).to_dicts() == [
        dict(cluster=1, chain_id=1),
        dict(cluster=1, chain_id=2),
    ]
    assert mmc.select_entity_clusters(db).to_dicts() == [
        dict(cluster=1, entity_id=1),
        dict(cluster=1, entity_id=2),
    ]


