import macromol_census as mmc
import polars as pl
import polars.testing

from test_database_io import insert_1abc

def test_ingest_blacklist(tmp_path):
    blacklist_path = tmp_path / 'blacklist.txt'
    blacklist_path.write_text('# comment\n1abc')

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    mmc.ingest_blacklist(db, blacklist_path)

    actual_blacklist = mmc.select_blacklisted_structures(db)
    expected_blacklist = pl.DataFrame([
        dict(struct_id=mmc.select_structure_id(db, '1abc'))
    ])

    pl.testing.assert_frame_equal(
            actual_blacklist,
            expected_blacklist,
            check_dtypes=False,
    )
