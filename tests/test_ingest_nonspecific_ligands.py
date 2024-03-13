import macromol_census as mmc
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
