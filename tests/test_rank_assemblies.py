import polars as pl
import macromol_census as mmc
import parametrize_from_file as pff

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


def test_rank_assemblies():
    db = mmc.open_db(':memory:')
    #db = mmc.open_db('foo.duckdb')
    mmc.init_db(db)

    # `1abc` contains an assembly to exercise each filtering/ranking rule.
    mmc.insert_structure(
            db, '1abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                # Eliminated: type isn't biologically relevant
                dict(id='1', type='point asymmetric unit', polymer_count=4),

                # Ranked 2nd: smaller than '6'
                dict(id='2', type='author_defined_assembly', polymer_count=4),

                # Eliminated: same subchains as '2', but smaller
                dict(id='3', type='author_defined_assembly', polymer_count=2),

                # Eliminated: not part of subchain cover
                dict(id='4', type='author_defined_assembly', polymer_count=1),

                # Ranked 3rd: same size as '2', but comes after
                dict(id='5', type='author_defined_assembly', polymer_count=4),

                # Ranked 1st: bigger than '2' and '5'
                dict(id='6', type='author_defined_assembly', polymer_count=6),
            ]),
            assembly_subchains=pl.DataFrame([
                dict(assembly_id='1', subchain_id='A'),
                dict(assembly_id='1', subchain_id='B'),
                dict(assembly_id='2', subchain_id='A'),
                dict(assembly_id='2', subchain_id='B'),
                dict(assembly_id='3', subchain_id='B'), # Flip order rel. to 2
                dict(assembly_id='3', subchain_id='A'), #
                dict(assembly_id='4', subchain_id='A'),
                dict(assembly_id='5', subchain_id='C'),
                dict(assembly_id='5', subchain_id='D'),
                dict(assembly_id='6', subchain_id='E'),
                dict(assembly_id='6', subchain_id='F'),
            ]),
            subchains=pl.DataFrame([
                dict(id='A', chain_id='A', entity_id='1'),
                dict(id='B', chain_id='B', entity_id='1'),
                dict(id='C', chain_id='C', entity_id='1'),
                dict(id='D', chain_id='D', entity_id='1'),
                dict(id='E', chain_id='E', entity_id='1'),
                dict(id='F', chain_id='F', entity_id='1'),
            ]),
            entities=pl.DataFrame([
                dict(id='1', type='polymer', formula_weight_Da=None),
            ]),
            polymer_entities=pl.DataFrame([
                dict(entity_id='1', type=None, sequence=None),
            ]),
    )

    # `2abc` is just to make sure there's no cross-talk between structures.
    mmc.insert_structure(
            db, '2abc',
            exptl_methods=[],
            deposit_date=None,
            full_atom=True,

            assemblies=pl.DataFrame([
                dict(id='1', type='author_defined_assembly', polymer_count=1),
            ]),
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

    ranks = mmc.rank_assemblies(db)
    mmc.insert_assembly_ranks(db, ranks)

    expected = unordered([
            dict(assembly_id=6, rank=1),
            dict(assembly_id=2, rank=2),
            dict(assembly_id=5, rank=3),
            dict(assembly_id=7, rank=1),
    ])

    assert ranks.to_dicts() == expected
    assert mmc.select_assembly_ranks(db).to_dicts() == expected

@pff.parametrize(
        schema=pff.cast(assembly_subchain=assembly_subchain),
)
def test_find_assembly_subchain_cover(assembly_subchain, expected):
    cover = mmc.find_assembly_subchain_cover(assembly_subchain)
    actual = list(cover['assembly_id'])

    assert any(
            actual == unordered(x.split())
            for x in expected
    )

