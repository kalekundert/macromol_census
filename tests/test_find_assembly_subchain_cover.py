import polars as pl
import macromol_census as mmc
import macromol_census.find_assembly_subchain_cover as mmc_
import parametrize_from_file as pff

from test_database_io import insert_1abc, insert_9xyz
from pytest_unordered import unordered

with_py = pff.Namespace()

def assembly_subchain(df_str):
    return dataframe(
            df_str,
            schema={'assembly_id': str, 'subchain_id': str},
    )

def dataframe(df_rows, schema):
    rows = []

    for row_str in df_rows:
        row = {k: f(x) for (k, f), x in zip(schema.items(), row_str.split())}
        rows.append(row)

    return pl.DataFrame(rows, schema)


def test_insert_assembly_subchain_covers():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    insert_1abc(db)
    insert_9xyz(db)

    mmc.insert_assembly_subchain_covers(db)

    # This isn't a very interesting example, because it's just all of the 
    # assemblies.  But `test_find_assembly_subchain_cover()` is meant to test 
    # the actual set cover algorithm; this is just meant to test the database 
    # IO.
    assert mmc.select_assembly_subchain_covers(db).to_dicts() == unordered([
            dict(assembly_id=1),
            dict(assembly_id=2),
            dict(assembly_id=3),
    ])

@pff.parametrize(
        schema=pff.cast(assembly_subchain=assembly_subchain),
)
def test_find_assembly_subchain_cover(assembly_subchain, expected):
    cover = mmc_._find_assembly_subchain_cover(assembly_subchain)
    actual = list(cover['assembly_id'])

    assert any(
            actual == unordered(x.split())
            for x in expected
    )

