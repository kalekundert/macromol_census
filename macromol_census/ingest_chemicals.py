"""\
Usage:
    mmc_ingest_chemicals <in:db> <in:chem_comp>

Arguments:
    <in:db>
        The path to a database created by `mmc_init`, and previously populated 
        by `mmc_ingest_structures`.

    <in:chem_comp>
        The path to a mmCIF file containing all the chemical components present 
        in the PDB.
"""

import polars as pl

from .util import extract_dataframe
from .database_io import open_db, insert_chemical_components
from gemmi.cif import read as read_cif
from tqdm import tqdm

def main():
    import docopt

    args = docopt.docopt(__doc__)
    db = open_db(args['<in:db>'])
    cif = read_cif(args['<in:chem_comp>'])

    def cif_wrapper(cif):
        for block in (progress_bar := tqdm(cif)):
            progress_bar.set_description(block.name)
            yield block

    ingest_chemical_components(db, cif_wrapper(cif))

def ingest_chemical_components(db, cif):
    df = pl.DataFrame(
            extract_chemical_component(block)
            for block in cif
    )
    insert_chemical_components(db, df)

def extract_chemical_component(block):
    chem_comp_descriptor = (
            extract_dataframe(
                block, 'pdbx_chem_comp_descriptor',
                required_cols=['type', 'descriptor'],
            )
    )
    return {
            'id': block.name,
            'inchi': get_descriptor(chem_comp_descriptor, 'InChI'),
            'inchi_key': get_descriptor(chem_comp_descriptor, 'InChIKey'),
    }

def get_descriptor(chem_comp_descriptor, type):
    # There are some chemical components without InChI strings, e.g. ASX.

    # I tried to use RDKit to create an InChI string from any of the SMILES 
    # strings given for this molecule, and wasn't able to.  So I think it's 
    # reasonable to give up and return null in this case.

    df = (
            chem_comp_descriptor
            .filter(pl.col('type') == type)
            .select('descriptor')
    )
    if df.is_empty():
        return None
    else:
        return df.item()

