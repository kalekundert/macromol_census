import polars as pl
import macromol_census as mmc
import macromol_census.ingest_mmcif as mmci
import parametrize_from_file as pff

from pytest_unordered import unordered
from polars.testing import assert_frame_equal
from functools import partial
from pathlib import Path
from datetime import date

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


def test_find_uningested_paths():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.insert_model(
            db,
            '9xyz',
            exptl_methods=['X-RAY DIFFRACTION'],
            deposit_date=date(year=2024, month=2, day=16),
            num_atoms=1000,
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

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_models(db, [CIF_DIR / '4erd.cif.gz'])

    assert mmc.select_models(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='4erd',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2012, month=4, day=19),
                num_atoms=2787,
            ),
    ]
    assert_frame_equal(
            mmc.select_assemblies(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
                dict(id=2, model_id=1, pdb_id='2'),
            ]),
    )
    assert_frame_equal(
            mmc.select_chains(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='A'),
                dict(id=2, model_id=1, pdb_id='B'),
                dict(id=3, model_id=1, pdb_id='C'),
                dict(id=4, model_id=1, pdb_id='D'),
            ]),
    )
    assert_frame_equal(
            mmc.select_entities(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
                dict(id=2, model_id=1, pdb_id='2'),
                dict(id=3, model_id=1, pdb_id='3'),
                dict(id=4, model_id=1, pdb_id='4'),
            ]),
    )
    assert_frame_equal(
            mmc.select_assembly_chain_pairs(db),
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
            mmc.select_chain_entity_pairs(db),
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
            mmc.select_entity_polymers(db),
            pl.DataFrame([
                dict(
                    entity_id=1,
                    type='polypeptide(L)',
                    sequence='MHHHHHHSIKQNCLIKIINIPQGTLKAEVVLAVRHLGYEFYCDYIDGQAMIRFQNSDEQRLAIQKLLNHNNNKLQIEIRGQICDVISTIPEDEEKNYWNYIKFKKNEFRKFFFMKKQQKKQNITQNYNK',
                ),
                dict(
                    entity_id=2,
                    type='polyribonucleotide',
                    sequence='GGUCGACAUCUUCGGAUGGACC',
                ),
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmc.select_entity_nonpolymers(db),
            pl.DataFrame([
                dict(entity_id=3, pdb_comp_id='K'),
                dict(entity_id=4, pdb_comp_id='HOH'),
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmc.select_qualities_xtal(db),
            pl.DataFrame([
                dict(model_id=1, resolution_A=2.589, num_reflections=16940, r_free=0.2717, r_work=0.2197),
            ]),
    )

    assert mmc.select_qualities_nmr(db).is_empty()
    assert mmc.select_qualities_em(db).is_empty()

def test_ingest_model_6wiv():
    # 6wiv causes problems because it specifies `_refine.ls_d_res_high` as `.`.  
    # Somehow this ends up getting interpreted as 0, which give incorrect 
    # results when filtering for the lowest resolution structures.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_models(db, [CIF_DIR / '6wiv.cif.gz'])

    assert mmc.select_models(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='6wiv',
                exptl_methods=['ELECTRON MICROSCOPY'],
                deposit_date=date(year=2020, month=4, day=10),
                num_atoms=11206,
            ),
    ]
    assert_frame_equal(
            mmc.select_assemblies(db),
            pl.DataFrame([
                dict(id=1, model_id=1, pdb_id='1'),
            ]),
    )

    # Manually determined subchain/chain/entity map:
    #
    # Subchain  Chain  Entity
    # ========  =====  ======
    # A         A      1
    # B         B      2
    # C         A      3
    # D         A      3
    # E         A      3
    # F         A      4
    # G         A      5
    # H         A      6
    # I         A      6
    # J         A      6
    # K         A      6
    # L         B      3
    # M         B      7
    # N         B      6
    # O         B      6
    # P         B      6
    # Q         B      6
    # R         B      6
    # S         B      6

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
                dict(id=3, model_id=1, pdb_id='3'),
                dict(id=4, model_id=1, pdb_id='4'),
                dict(id=5, model_id=1, pdb_id='5'),
                dict(id=6, model_id=1, pdb_id='6'),
                dict(id=7, model_id=1, pdb_id='7'),
                ]),
            )
    assert_frame_equal(
            mmc.select_assembly_chain_pairs(db),
            pl.DataFrame([
                dict(assembly_id=1, chain_id=1),
                dict(assembly_id=1, chain_id=2),
                ]),
            check_row_order=False,
            )
    assert_frame_equal(
            # Each protein subunit is binding a different lipid (U3G and U3D, 
            # respectively).  Furthermore, one subunit is also binding a 
            # calcium ion.  The other ligands are not specifically bound, and 
            # are present in both chains.

            mmc.select_chain_entity_pairs(db),
            pl.DataFrame([
                dict(chain_id=1, entity_id=1), # protein
                dict(chain_id=1, entity_id=3), # resn NAG
                dict(chain_id=1, entity_id=4), # resn CA
                dict(chain_id=1, entity_id=5), # resn U3G
                dict(chain_id=1, entity_id=6), # resn CLR

                dict(chain_id=2, entity_id=2), # protein
                dict(chain_id=2, entity_id=3), # resn NAG
                dict(chain_id=2, entity_id=6), # resn CLR
                dict(chain_id=2, entity_id=7), # resn U3D
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmc.select_entity_polymers(db),
            pl.DataFrame([
                dict(
                    entity_id=1,
                    type='polypeptide(L)',
                    sequence='MGPGAPFARVGWPLPLLVVMAAGVAPVWASHSPHLPRPHSRVPPHPSSERRAVYIGALFPMSGGWPGGQACQPAVEMALEDVNSRRDILPDYELKLIHHDSKCDPGQATKYLYELLYNDPIKIILMPGCSSVSTLVAEAARMWNLIVLSYGSSSPALSNRQRFPTFFRTHPSATLHNPTRVKLFEKWGWKKIATIQQTTEVFTSTLDDLEERVKEAGIEITFRQSFFSDPAVPVKNLKRQDARIIVGLFYETEARKVFCEVYKERLFGKKYVWFLIGWYADNWFKIYDPSINCTVDEMTEAVEGHITTEIVMLNPANTRSISNMTSQEFVEKLTKRLKRHPEETGGFQEAPLAYDAIWALALALNKTSGGGGRSGVRLEDFNYNNQTITDQIYRAMNSSSFEGVSGHVVFDASGSRMAWTLIEQLQGGSYKKIGYYDSTKDDLSWSKTDKWIGGSPPADQTLVIKTFRFLSQKLFISVSVLSSLGIVLAVVCLSFNIYNSHVRYIQNSQPNLNNLTAVGCSLALAAVFPLGLDGYHIGRNQFPFVCQARLWLLGLGFSLGYGSMFTKIWWVHTVFTKKEEKKEWRKTLEPWKLYATVGLLVGMDVLTLAIWQIVDPLHRTIETFAKEEPKEDIDVSILPQLEHCSSRKMNTWLGIFYGYKGLLLLLGIFLAYETKSVSTEKINDHRAVGMAIYNVAVLCLITAPVTMILSSQQDAAFAFASLAIVFSSYITLVVLFVPKMRRLITRGEWQSEAQDTMKTGSSTNNNEEEKSRLLEKENRELEKIIAEKEERVSELRHQLQSRDYKDDDDK',
                ),
                dict(
                    entity_id=2,
                    type='polypeptide(L)',
                    sequence='MASPRSSGQPGPPPPPPPPPARLLLLLLLPLLLPLAPGAWGWARGAPRPPPSSPPLSIMGLMPLTKEVAKGSIGRGVLPAVELAIEQIRNESLLRPYFLDLRLYDTECDNAKGLKAFYDAIKYGPNHLMVFGGVCPSVTSIIAESLQGWNLVQLSFAATTPVLADKKKYPYFFRTVPSDNAVNPAILKLLKHYQWKRVGTLTQDVQRFSEVRNDLTGVLYGEDIEISDTESFSNDPCTSVKKLKGNDVRIILGQFDQNMAAKVFCCAYEENMYGSKYQWIIPGWYEPSWWEQVHTEANSSRCLRKNLLAAMEGYIGVDFEPLSSKQIKTISGKTPQQYEREYNNKRSGVGPSKFHGYAYDGIWVIAKTLQRAMETLHASSRHQRIQDFNYTDHTLGRIILNAMNETNFFGVTGQVVFRNGERMGTIKFTQFQDSREVKVGEYNAVADTLEIINDTIRFQGSEPPKDKTIILEQLRKISLPLYSILSALTILGMIMASAFLFFNIKNRNQKLIKMSSPYMNNLIILGGMLSYASIFLFGLDGSFVSEKTFETLCTVRTWILTVGYTTAFGAMFAKTWRVHAIFKNVKMKKKIIKDQKLLVIVGGMLLIDLCILICWQAVDPLRRTVEKYSMEPDPAGRDISIRPLLEHCENTHMTIWLGIVYAYKGLLMLFGCFLAWETRNVSIPALNDSKYIGMSVYNVGIMCIIGAAVSFLTRDQPNVQFCIVALVIIFCSTITLCLVFVPKLITLRTNPDAATQNRRFQFTQNQKKEDSKTSTSVTSVNQASTSRLEGLQSENHRLRMKITELDKDLEEVTMQLQDTDYKDDDDK',
                ),
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmc.select_entity_nonpolymers(db),
            pl.DataFrame([
                dict(entity_id=3, pdb_comp_id='NAG'),
                dict(entity_id=4, pdb_comp_id='CA'),
                dict(entity_id=5, pdb_comp_id='U3G'),
                dict(entity_id=6, pdb_comp_id='CLR'),
                dict(entity_id=7, pdb_comp_id='U3D'),
            ]),
            check_row_order=False,
    )
    assert_frame_equal(
            mmc.select_qualities_em(db),
            pl.DataFrame([
                dict(model_id=1, resolution_A=3.3, q_score=None),
            ]),
    )

    assert mmc.select_qualities_xtal(db).is_empty()
    assert mmc.select_qualities_nmr(db).is_empty()

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

    actual = mmci._make_chain_subchain_entity_id_map(atom_site)
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
    cover = mmci._find_covering_assemblies(assembly_subchain_pairs)
    actual = list(cover['assembly_id'])

    debug(assembly_subchain_pairs, actual, expected)
    assert any(
            actual == unordered(x.split())
            for x in expected
    )
