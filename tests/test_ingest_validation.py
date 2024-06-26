import macromol_census as mmc
import macromol_census.ingest_validation as mmci

from gemmi.cif import read as read_cif
from pytest import approx
from pathlib import Path

CIF_DIR = Path(__file__).parent / 'pdb'

def test_extract_nmr_6dze():
    cif = read_cif(str(CIF_DIR / '6dze_validation.cif.gz')).sole_block()
    assert mmci._extract_nmr_restraints(cif) == 162

def test_extract_em_6dzp():
    cif = read_cif(str(CIF_DIR / '6dzp_validation.cif.gz')).sole_block()
    quality = mmci._extract_em_resolution_q_score(cif)
    assert quality == {
            'resolution_A': approx(3.42),  # EMDB resolution
            'q_score': approx(0.464),
    }
    
def test_extract_em_8dzr():
    # I originally included this test case because it has "calculated", as 
    # opposed to "author provided", FSC resolutions.  However, since then, I 
    # decided to use only the EMDB resolution, so the distinction doesn't 
    # really matter any more.  I kept the test case, though, in case I ever go 
    # back to the old behavior.

    cif = read_cif(str(CIF_DIR / '8dzr_validation.cif.gz')).sole_block()
    quality = mmci._extract_em_resolution_q_score(cif)
    assert quality == {
            'resolution_A': approx(2.61),
            'q_score': approx(0.485),
    }
    
def test_extract_em_6eri():
    # This validation file is unusual in that it specifies the string "None" 
    # for all the FSC resolutions.  I'm no longer using these resolutions, so 
    # this doesn't really matter anymore, but previously I had to account for 
    # this specially.
    cif = read_cif(str(CIF_DIR / '6eri_validation.cif.gz')).sole_block()
    quality = mmci._extract_em_resolution_q_score(cif)
    assert quality == {
            'resolution_A': approx(3),
            'q_score': approx(0.514),
    }
    
def test_extract_clashscore_6dze():
    cif = read_cif(str(CIF_DIR / '6dze_validation.cif.gz')).sole_block()
    assert mmci._extract_clashscore(cif) == approx(27.27)

def test_extract_clashscore_4iio():
    # This validation file specifies a clashscore of "-1.00".  I assume that 
    # this is some sort of error code, so I replace it with null.
    cif = read_cif(str(CIF_DIR / '4iio_validation.cif.gz')).sole_block()
    assert mmci._extract_clashscore(cif) is None

def test_ingest_validation_report_2wls():
    # This validation file doesn't properly specify a PDB ID, so instead this 
    # information has to be parsed from the file name.
    
    db = mmc.open_db(':memory:')
    mmc.init_db(db)
    db.executemany(
            'INSERT INTO structure (pdb_id, full_atom) VALUES (?, TRUE)',
            [('9xyz',), ('2wls',)],
    )
    db.sql('SELECT * FROM model').show()

    mmc.ingest_validation_report(db, CIF_DIR / '2wls_validation.cif.gz')

    assert mmc.select_clashscores(db).to_dicts() == [
            dict(
                struct_id=2,
                source='mmcif_pdbx_vrpt',
                clashscore=approx(11.03),
            ),
    ]
