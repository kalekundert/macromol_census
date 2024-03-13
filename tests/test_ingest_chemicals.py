import macromol_census as mmc

from gemmi.cif import read as read_cif
from pathlib import Path

def test_ingest_chemicals():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    cif = read_cif('structures/components.cif.gz')
    mmc.ingest_chemical_components(db, cif)

    assert mmc.select_chemical_components(db).to_dicts() == [
            dict(
                pdb_id='EQU',
                inchi='InChI=1S/C18H18O2/c1-18-9-8-14-13-5-3-12(19)10-11(13)2-4-15(14)16(18)6-7-17(18)20/h2-5,10,16,19H,6-9H2,1H3/t16-,18-/m0/s1',
                inchi_key='PDRGHUMCVRDZLQ-WMZOPIPTSA-N',
            ),
    ]

    
