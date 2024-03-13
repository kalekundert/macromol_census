"""\
Create an empty database.

Usage:
    mmc_init <out:db-path>

This database can be filled with information on all the structures, models, 
assemblies, chains, subchains, and entities in the PDB.  Collectively, this 
information can then be used to produce a set of assemblies with minimal 
redundancy.
"""

from .database_io import open_db, init_db

def main():
    import docopt

    args = docopt.docopt(__doc__)

    db = open_db(args['<out:db-path>'])
    init_db(db)
