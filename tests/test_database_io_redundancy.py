import macromol_census.database_io.redundancy as mmc
import polars as pl

from polars.testing import assert_frame_equal
from functools import partial

assert_frame_equal = partial(assert_frame_equal, check_dtype=False)

def test_model():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    quality = mmc.QualityXtal(
            resolution_A=2.9,
            reflections_per_atom=4.5,
            r_free=0.23,
            r_work=0.19,
    )
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
            quality=quality,
            assembly_chain_pairs=assembly_chain_pairs,
            chain_entity_pairs=chain_entity_pairs,
            polymers=polymers,
            nonpolymers=nonpolymers,
    )

    assert_frame_equal(
            mmc.select_models(db),
            pl.DataFrame([
                dict(id=1, pdb_id='9xyz'),
            ]),
    )
    assert_frame_equal(
            mmc.select_model_qualities_xtal(db),
            pl.DataFrame([
                dict(
                    model_id=1,
                    resolution_A=2.9,
                    reflections_per_atom=4.5,
                    r_free=0.23,
                    r_work=0.19,
                ),
            ]),
    )
    assert_frame_equal(
            mmc.select_assemblies(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
            ]),
    )
    assert_frame_equal(
            mmc.select_chains(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='A'),
                dict(id=2, model_id=1, pdb_id='B'),
            ]),
    )
    assert_frame_equal(
            mmc.select_entities(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
                dict(id=2, model_id=1, pdb_id='2'),
            ]),
    )
    assert_frame_equal(
            mmc.select_assembly_chain_pairs(db),
            pl.DataFrame([
                dict(assembly_id=1, chain_id=1),
                dict(assembly_id=1, chain_id=2),
            ]),
    )
    assert_frame_equal(
            mmc.select_chain_entity_pairs(db),
            pl.DataFrame([
                dict(chain_id=1, entity_id=1),
                dict(chain_id=1, entity_id=2),
                dict(chain_id=2, entity_id=1),
                dict(chain_id=2, entity_id=2),
            ]),
    )
