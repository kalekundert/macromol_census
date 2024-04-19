import macromol_census as mmc
import polars as pl

from test_database_io import insert_1abc, insert_9xyz

def test_ingest_nonspecific_ligands(tmp_path):
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    ignore_path = tmp_path / 'ignore'
    ignore_path.write_text('EQU\nABC\n')

    mmc.ingest_nonspecific_ligands(db, ignore_path)

    assert mmc.select_ignored_entities(db).to_dicts() == [
            dict(entity_id=3),
    ]

def test_ignore_low_weight_ligands():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type=None, polymer_count=0),
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
                dict(id='1', type='non-polymer', formula_weight_Da=49),
                dict(id='2', type='non-polymer', formula_weight_Da=51),
            ]),
            monomer_entities=pl.DataFrame([
                dict(entity_id='1', comp_id='ABC'),
                dict(entity_id='2', comp_id='XYZ'),
            ]),
    )

    mmc.ignore_low_weight_ligands(db, 50)

    assert mmc.select_ignored_entities(db).to_dicts() == [
            dict(entity_id=1),
    ]

