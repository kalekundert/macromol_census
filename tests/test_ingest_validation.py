import macromol_census.ingest_validation as mmc

from gemmi.cif import read as read_cif
from pytest import approx
from pathlib import Path

CIF_DIR = Path(__file__).parent / 'structures'

def test_extract_nmr_6dze():
    cif = read_cif(str(CIF_DIR / '6dze_validation.cif.gz')).sole_block()
    assert mmc._extract_nmr_restraints(cif) == 162

def test_extract_em_6dzp():
    cif = read_cif(str(CIF_DIR / '6dzp_validation.cif.gz')).sole_block()
    quality = mmc._extract_em_resolution_q_score(cif)
    assert quality == {
            'resolution_A': approx(3.40),  # FSC with 0.143 cutoff
            'q_score': approx(0.464),
    }
    
def test_extract_em_8dzr():
    # 8dzr has a "calculated" resolution, while 6dzp has a "user specified" 
    # resolution.  Need to check for both.

    cif = read_cif(str(CIF_DIR / '8dzr_validation.cif.gz')).sole_block()
    quality = mmc._extract_em_resolution_q_score(cif)
    assert quality == {
            'resolution_A': approx(3.09),
            'q_score': approx(0.485),
    }
    
def test_extract_em_6eri():
    # This validation file is unusual in that it specifies the string "None" as 
    # a resolution.  This has to be handled specially.
    cif = read_cif(str(CIF_DIR / '6eri_validation.cif.gz')).sole_block()
    quality = mmc._extract_em_resolution_q_score(cif)
    assert quality == {
            'resolution_A': approx(3),
            'q_score': approx(0.514),
    }
    
def test_extract_clashscore_6dze():
    cif = read_cif(str(CIF_DIR / '6dze_validation.cif.gz')).sole_block()
    assert mmc._extract_clashscore(cif) == approx(27.27)
