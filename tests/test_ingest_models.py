import polars as pl
import macromol_census.ingest_models as mmc
import macromol_census.database_io.redundancy as mmcdb
import parametrize_from_file as pff

from macromol_census.database_io.redundancy import (
        open_db, init_db, insert_model, QualityXtal,
)
from pytest_unordered import unordered
from polars.testing import assert_frame_equal
from functools import partial
from pathlib import Path

CIF_DIR = Path(__file__).parent / 'structures'

with_py = pff.Namespace()
assert_frame_equal = partial(assert_frame_equal, check_dtype=False)

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

    debug

    return pl.DataFrame(rows, schema)

def quality_xtal(x):
    return mmcdb.QualityXtal(**with_py.eval(x))


def test_find_uningested_paths():
    db = mmcdb.open_db(':memory:')
    mmcdb.init_db(db)

    mmcdb.insert_model(
            db, '9xyz',
            assembly_chain_pairs=pl.DataFrame([
                dict(assembly_id='1', chain_id='A'),
            ]),
            chain_entity_pairs=pl.DataFrame([
                dict(chain_id='A', entity_id='1'),
            ]),
    )

    uningested_paths = mmc.find_uningested_paths(
            db,
            cif_paths=['1abc', '9xyz'],
            pdb_id_from_path=lambda x: x,
    )

    assert uningested_paths == ['1abc']

def test_ingest_model_4erd():
    # 4erd is an interesting model, because it's one of the few examples in the 
    # PDB where a single chain (an RNA double helix, in this case) appears in 
    # multiple biological assemblies.

    db = mmcdb.open_db(':memory:')
    mmcdb.init_db(db)

    mmc._ingest_model(db, CIF_DIR / '4erd.cif')

    assert_frame_equal(
            mmcdb.select_models(db),
            pl.DataFrame([
                dict(id=1, pdb_id='4erd'),
            ]),
    )
    assert_frame_equal(
            mmcdb.select_assemblies(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
                dict(id=2, model_id=1, pdb_id='2'),
            ]),
    )
    assert_frame_equal(
            mmcdb.select_chains(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='A'),
                dict(id=2, model_id=1, pdb_id='B'),
                dict(id=3, model_id=1, pdb_id='C'),
                dict(id=4, model_id=1, pdb_id='D'),
            ]),
    )
    assert_frame_equal(
            mmcdb.select_entities(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
                dict(id=2, model_id=1, pdb_id='2'),
                dict(id=3, model_id=1, pdb_id='3'),
                dict(id=4, model_id=1, pdb_id='4'),
            ]),
    )
    debug(
            mmcdb.select_assembly_chain_pairs(db),
    )
    assert_frame_equal(
            mmcdb.select_assembly_chain_pairs(db),
            pl.DataFrame([
                dict(assembly_id=1, chain_id=1),  # protein
                dict(assembly_id=1, chain_id=3),  # RNA
                dict(assembly_id=1, chain_id=4),  # RNA

                dict(assembly_id=2, chain_id=2),  # protein
                dict(assembly_id=2, chain_id=3),  # RNA
                dict(assembly_id=2, chain_id=4),  # RNA
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmcdb.select_chain_entity_pairs(db),
            pl.DataFrame([
                dict(chain_id=1, entity_id=1),  # protein
                dict(chain_id=1, entity_id=3),  # potassium
                dict(chain_id=1, entity_id=4),  # water

                dict(chain_id=2, entity_id=1),  # protein
                dict(chain_id=2, entity_id=4),  # water

                dict(chain_id=3, entity_id=2),  # RNA
                dict(chain_id=3, entity_id=4),  # water

                dict(chain_id=4, entity_id=2),  # RNA
                dict(chain_id=4, entity_id=4),  # water
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmcdb.select_entity_polymers(db),
            pl.DataFrame([
                dict(
                    entity_id=1,
                    type='polypeptide(L)',
                    sequence='XHHHHHHSIKQNCLIKIINIPQGTLKAEVVLAVRHLGYEFYCDYIDGQAXIRFQNSDEQRLAIQKLLNHNNNKLQIEIRGQICDVISTIPEDEEKNYWNYIKFKKNEFRKFFFXKKQQKKQNITQNYNK',
                ),
                dict(
                    entity_id=2,
                    type='polyribonucleotide',
                    sequence='GGUCGACAUCUUCGGAUGGACC',
                ),
            ]),
            check_row_order=False,
    )
    debug(
            mmcdb.select_entity_nonpolymers(db),
    )
    assert_frame_equal(
            mmcdb.select_entity_nonpolymers(db),
            pl.DataFrame([
                dict(entity_id=3, pdb_comp_id='K'),
                dict(entity_id=4, pdb_comp_id='HOH'),
            ]),
            check_row_order=False,
    )


@pff.parametrize(
        schema=pff.cast(expected=quality_xtal),
        indirect=['tmp_files'],
)
def test_parse_quality(tmp_files, expected):
    actual = mmc._parse_quality(tmp_files / 'data.json')
    assert actual == expected

def test_make_chain_subchain_entity_map():
    atom_site = pl.DataFrame([
        dict(auth_asym_id='AAA', label_asym_id='A', label_entity_id='1'),
        dict(auth_asym_id='AAA', label_asym_id='A', label_entity_id='1'),

        dict(auth_asym_id='AAA', label_asym_id='B', label_entity_id='2'),
        dict(auth_asym_id='AAA', label_asym_id='B', label_entity_id='2'),

        dict(auth_asym_id='AAA', label_asym_id='C', label_entity_id='3'),
        dict(auth_asym_id='AAA', label_asym_id='C', label_entity_id='3'),

        dict(auth_asym_id='BBB', label_asym_id='D', label_entity_id='1'),
        dict(auth_asym_id='BBB', label_asym_id='D', label_entity_id='1'),

        dict(auth_asym_id='BBB', label_asym_id='E', label_entity_id='2'),
        dict(auth_asym_id='BBB', label_asym_id='E', label_entity_id='2'),

        dict(auth_asym_id='BBB', label_asym_id='F', label_entity_id='3'),
        dict(auth_asym_id='BBB', label_asym_id='F', label_entity_id='3'),
    ])

    actual = mmc._make_chain_subchain_entity_id_map(atom_site)
    expected = pl.DataFrame([
        dict(chain_id='AAA', subchain_id='A', entity_id='1'),
        dict(chain_id='AAA', subchain_id='B', entity_id='2'),
        dict(chain_id='AAA', subchain_id='C', entity_id='3'),
        dict(chain_id='BBB', subchain_id='D', entity_id='1'),
        dict(chain_id='BBB', subchain_id='E', entity_id='2'),
        dict(chain_id='BBB', subchain_id='F', entity_id='3'),
    ])

    assert_frame_equal(actual, expected, check_row_order=False)

@pff.parametrize(
        schema=pff.cast(
            assembly_subchain_pairs=assembly_subchain,
        ),
)
def test_find_covering_assemblies(assembly_subchain_pairs, expected):
    cover = mmc._find_covering_assemblies(assembly_subchain_pairs)
    actual = list(cover['assembly_id'])

    debug(assembly_subchain_pairs, actual, expected)
    assert any(
            actual == unordered(x.split())
            for x in expected
    )
