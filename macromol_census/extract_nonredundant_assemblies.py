"""\
Usage:
    mmc_extract_nonredundant_assemblies <in:db> [<out:path>]

Arguments:
    <in:db>
        A database created by `mmc_init` and populated by 
        `mmc_pick_assemblies`.

    <out:path>
        The path where the extracted assemblies should be written to.  If not 
        path is specified, each assembly will be printed to stdout in JSON 
        form.
"""

import json, sys
from .database_io import open_db

def main():
    import docopt

    args = docopt.docopt(__doc__)
    db = open_db(args['<in:db>'])
    out_path = args['<out:path>']

    assemblies = select_nonredundant_pdb_ids(db)

    if out_path is not None:
        assemblies.write_parquet(out_path)
    else:
        try:
            for row in assemblies.iter_rows(named=True):
                json.dump(row, sys.stdout)
                sys.stdout.write('\n')
        except BrokenPipeError:
            pass

def select_nonredundant_pdb_ids(db):
    pdb_subchain = db.sql('''\
            SELECT 
                assembly_subchain.assembly_id AS assembly_id,
                list(subchain.pdb_id) AS pdb_subchain_ids
            FROM nonredundant
            JOIN assembly_subchain USING (subchain_id)
            JOIN subchain ON subchain.id = assembly_subchain.subchain_id
            GROUP BY assembly_id
    ''')
    pdb_subchain_pair = db.sql('''\
            SELECT 
                assembly_subchain_1.assembly_id AS assembly_id,
                list([subchain_1.pdb_id, subchain_2.pdb_id]) AS pdb_subchain_id_pairs
            FROM nonredundant_pair
            JOIN assembly_subchain AS assembly_subchain_1
                ON assembly_subchain_1.subchain_id = nonredundant_pair.subchain_id_1
            JOIN assembly_subchain AS assembly_subchain_2
                ON assembly_subchain_2.subchain_id = nonredundant_pair.subchain_id_2
            JOIN subchain AS subchain_1
                ON subchain_1.id = assembly_subchain_1.subchain_id
            JOIN subchain AS subchain_2
                ON subchain_2.id = assembly_subchain_2.subchain_id
            GROUP BY assembly_subchain_1.assembly_id
    ''')
    pdb_assembly = db.sql('''\
            SELECT
                coalesce(
                    pdb_subchain.assembly_id, 
                    pdb_subchain_pair.assembly_id, 
                ) AS assembly_id,
                pdb_subchain.pdb_subchain_ids AS pdb_subchain_ids,
                pdb_subchain_pair.pdb_subchain_id_pairs AS pdb_subchain_id_pairs, 
            FROM pdb_subchain
            FULL JOIN pdb_subchain_pair USING (assembly_id)
    ''')
    pdb_structure = db.sql('''\
            SELECT
                structure.pdb_id AS pdb_id,
                structure.rank,
                -- first(model.pdb_id) AS model,
                assembly.pdb_id AS assembly,
                pdb_assembly.pdb_subchain_ids AS subchains,
                pdb_assembly.pdb_subchain_id_pairs AS subchain_pairs
            FROM pdb_assembly
            JOIN assembly ON assembly.id = pdb_assembly.assembly_id
            JOIN structure ON structure.id = assembly.struct_id
            -- JOIN model ON structure.id = model.struct_id
            GROUP BY ALL
            ORDER BY structure.rank
    ''')
    return pdb_structure.pl()





