import polars as pl
import macromol_census as mmc

from itertools import zip_longest
from datetime import date

def insert_quality_metrics(
        db,
        pdb_id, 
        deposit_date=None,
        xtal_resolutions=[],
        xtal_r_frees=[],
        em_resolutions=[],
        em_q_scores=[],
        nmr_dist_restraints=[],
        clashscores=[],
):
    struct_id = mmc.insert_structure(
            db, pdb_id,
            exptl_methods=[],
            deposit_date=deposit_date,
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
                dict(entity_id='1', type='polypeptide(L)', sequence=None),
            ]),
            xtal_quality=pl.DataFrame({
                'resolution_A': xtal_resolutions,
                'r_free': xtal_r_frees or [None] * len(xtal_resolutions),
                'r_work': [None] * len(xtal_resolutions),
            }),
    )

    for em_resolution, em_q_score in zip_longest(em_resolutions, em_q_scores):
        mmc.insert_em_quality(
                db, struct_id,
                source='mmcif_pdbx_vrpt',
                resolution_A=em_resolution,
                q_score=em_q_score,
        )

    for n in nmr_dist_restraints:
        mmc.insert_nmr_quality(
                db, struct_id,
                source='mmcif_pdbx_vrpt',
                num_dist_restraints=n,
        )

    for clashscore in clashscores:
        mmc.insert_clashscore(
                db, struct_id,
                source='mmcif_pdbx_vrpt',
                clashscore=clashscore,
        )


def test_rank_assemblies_resolution():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_quality_metrics(
            db, '5abc',
            em_resolutions=[3.0],
    )
    insert_quality_metrics(
            db, '1abc',
            xtal_resolutions=[1.0],
    )
    insert_quality_metrics(
            # If multiple resolutions, use lowest.  Also note that EM and 
            # crystal resolutions are treated identically (unless both are 
            # present in the same structure).
            db, '2abc',
            em_resolutions=[1.5, 5.0],
    )
    insert_quality_metrics( 
            # Prefer crystal over EM, if both present.
            db, '3abc',
            xtal_resolutions=[2.0],
            em_resolutions=[0.5],
    )
    insert_quality_metrics( 
            db, '4abc',
            xtal_resolutions=[2.5, 6.0],
    )

    assert mmc.rank_structures(db).to_dicts() == [
            dict(struct_id=2, rank=1),
            dict(struct_id=3, rank=2),
            dict(struct_id=4, rank=3),
            dict(struct_id=5, rank=4),
            dict(struct_id=1, rank=5),
    ]

def test_rank_assemblies_clashscore():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # Resolutions vary, but are all in the same bin.

    insert_quality_metrics( 
            db, '3abc',
            em_resolutions=[2.96],
            clashscores=[3.0],
    )
    insert_quality_metrics(
            db, '1abc',
            xtal_resolutions=[3.00],
            clashscores=[1.0],
    )
    insert_quality_metrics(
            # If multiple scores, use the best (lowest) one.
            db, '2abc',
            xtal_resolutions=[3.04],
            clashscores=[2.0, 4.0],
    )

    assert mmc.rank_structures(db).to_dicts() == [
            dict(struct_id=2, rank=1),
            dict(struct_id=3, rank=2),
            dict(struct_id=1, rank=3),
    ]

def test_rank_assemblies_quality():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    # Clashscores vary, but are all in the same bin.

    insert_quality_metrics( 
            # Resolutions above 4Å are treated as unspecified.
            db, '9abc',
            clashscores=[1.96],
            em_resolutions=[4.1],
            em_q_scores=[0.7],
    )
    insert_quality_metrics( 
            db, '1abc',
            clashscores=[1.97],
            nmr_dist_restraints=[4000],
    )
    insert_quality_metrics( 
            db, '2abc',
            clashscores=[1.98],
            nmr_dist_restraints=[1000, 3000],
    )
    insert_quality_metrics( 
            db, '3abc',
            clashscores=[1.99],
            nmr_dist_restraints=[2000],
    )
    insert_quality_metrics( 
            db, '4abc',
            clashscores=[2.00],
            xtal_resolutions=[4.1],
            xtal_r_frees=[0.2],
    )
    insert_quality_metrics( 
            db, '5abc',
            clashscores=[2.01],
            xtal_resolutions=[4.1, 4.1],
            xtal_r_frees=[0.3, 0.5],
    )
    insert_quality_metrics( 
            db, '6abc',
            clashscores=[2.02],
            xtal_resolutions=[4.1],
            xtal_r_frees=[0.4],
    )
    insert_quality_metrics( 
            db, '7abc',
            clashscores=[2.03],
            em_resolutions=[4.1],
            em_q_scores=[0.9],
    )
    insert_quality_metrics( 
            db, '8abc',
            clashscores=[2.04],
            em_resolutions=[4.1, 4.1],
            em_q_scores=[0.8, 0.6],
    )

    assert mmc.rank_structures(db).to_dicts() == [
            dict(struct_id=2, rank=1),
            dict(struct_id=3, rank=2),
            dict(struct_id=4, rank=3),
            dict(struct_id=5, rank=4),
            dict(struct_id=6, rank=5),
            dict(struct_id=7, rank=6),
            dict(struct_id=8, rank=7),
            dict(struct_id=9, rank=8),
            dict(struct_id=1, rank=9),
    ]

def test_rank_assemblies_tiebreakers():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_quality_metrics( 
            db, '1abc',
            deposit_date=date(2024, 3, 11),
    )
    insert_quality_metrics( 
            db, '3abc',
            deposit_date=date(2024, 3, 12),
    )
    insert_quality_metrics( 
            db, '2abc',
            deposit_date=date(2024, 3, 11),
    )

    assert mmc.rank_structures(db).to_dicts() == [
            dict(struct_id=2, rank=1),
            dict(struct_id=3, rank=2),
            dict(struct_id=1, rank=3),
    ]
