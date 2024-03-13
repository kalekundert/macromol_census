"""
Cluster ligands by identity.

Usage:
    mmc_find_identical_ligands <in:db>
"""

from .database_io import open_db, insert_entity_clusters

def main():
    import docopt
    args = docopt.docopt(__doc__)

    db = open_db(args['<in:db>'])

    clusters = find_identical_ligands(db)
    insert_entity_clusters(db, clusters, 'identical-ligands')

def find_identical_ligands(db):
    return db.sql('''\
            SELECT
                entity_id,
                pdb_comp_id AS cluster_id
            FROM entity_monomer
    ''').pl()
