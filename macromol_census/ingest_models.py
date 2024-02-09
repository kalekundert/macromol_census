"""\
Usage:
    ingest_models <in:cif-dir> <out:db-path> [-x exclude]

Arguments:
    <dir>
        The path to a directory containing 

Options:
    -x --exclude <path>
        A path to a newline-separated list of PDB ids to exclude from the 
        dataset.
"""

import numpy as np
import polars as pl
import os.path
import json

from .database_io.redundancy import (
        open_db, init_db, select_models, insert_model, QualityXtal,
)
from .error import UsageError
from gemmi.cif import read_file as read_cif
from scipy.optimize import milp, Bounds, LinearConstraint

def main():
    import docopt
    from pathlib import Path
    from more_itertools import ilen
    from tqdm import tqdm

    args = docopt.docopt(__doc__)
    cif_dir = Path(args['<in:cif-dir>'])
    db = open_db(args['<out:db-path>'])

    init_db(db)
    cif_paths = find_uningested_paths(
            db,
            cif_paths=tqdm(
                cif_dir.glob('**/*.cif'),
                desc='find paths to ingest',
            ),
            pdb_id_from_path=lambda p: p.parent.name,
    )
    ingest_models(db, tqdm(cif_paths, desc='ingest models'))

def find_uningested_paths(db, cif_paths, pdb_id_from_path):
    already_ingested = set(select_models(db)['pdb_id'].unique())
    return [
            p for p in cif_paths
            if pdb_id_from_path(p) not in already_ingested
    ]

def ingest_models(db, cif_paths):
    try:
        for cif_path in cif_paths:
            _ingest_model(db, cif_path)

    except Exception as err1:
        err2 = UsageError(path=cif_path, cause=str(err1))
        err2.brief = "{cause}"
        err2.info += "path: {path}"
        raise err2 from err1

def _ingest_model(db, cif_path):
    cif = read_cif(str(cif_path)).sole_block()
    pdb_id = cif.name.lower()
    quality = _parse_quality(cif_path.parent / 'data.json')

    def extract_df(key, required_cols=None):
        loop = cif.get_mmcif_category(f'_{key}.')
        df = pl.DataFrame(loop)

        if required_cols:
            missing_cols = [x for x in required_cols if x not in df.columns]
            if missing_cols:
                err = UsageError(
                        path=cif_path,
                        category=key,
                        missing_cols=missing_cols,
                )
                err.brief = "missing required column(s)"
                err.info += "path: {path}"
                err.info += "category: _{category}.*"
                err.blame += "missing column(s): {missing_cols}"
                raise err

        return df

    atom_site = extract_df('atom_site')
    struct_assembly_gen = extract_df('pdbx_struct_assembly_gen')
    entity_poly = extract_df('entity_poly')
    entity_nonpoly = extract_df('pdbx_entity_nonpoly')

    id_map = _make_chain_subchain_entity_id_map(atom_site)
    assembly_chain_pairs = _find_covering_assembly_chain_pairs(
            struct_assembly_gen,
            id_map,
    )
    chain_entity_pairs = _find_chain_entity_pairs(id_map)
    polymers = _find_polymers(entity_poly)
    nonpolymers = _find_nonpolymers(entity_nonpoly)

    insert_model(
            db, pdb_id,
            quality=quality,
            assembly_chain_pairs=assembly_chain_pairs,
            chain_entity_pairs=chain_entity_pairs,
            polymers=polymers,
            nonpolymers=nonpolymers,
    )

def _parse_quality(json_path):
    if not json_path.exists() or os.path.getsize(json_path) == 0:
        return None

    with open(json_path) as f:
        meta = json.load(f)

    def get_property(meta, key):
        try:
            return meta['properties'][key]
        except KeyError:
            return None

    return QualityXtal(
            resolution_A=get_property(meta, 'RESOLUTION'),
            reflections_per_atom=get_property(meta, 'REFPATM'),
            r_work=get_property(meta, 'RFIN'),
            r_free=get_property(meta, 'RFFIN'),
    )

def _make_chain_subchain_entity_id_map(atom_site):
    return (
            atom_site
            .select(
                chain_id='auth_asym_id',
                subchain_id='label_asym_id',
                entity_id='label_entity_id',
            )
            .unique()
    )

def _find_covering_assembly_chain_pairs(struct_assembly_gen, id_map):
    if struct_assembly_gen.is_empty():
        assembly_chain = (
                id_map
                .select(
                    pl.lit('1').alias('assembly_id'),
                    'chain_id',
                )
        )
        return assembly_chain

    assembly_subchain = (
            struct_assembly_gen
            .select(
                'assembly_id',
                subchain_id=pl.col('asym_id_list').str.split(','),
            )
            .explode('subchain_id')
    )
    chain_subchain = (
            id_map
            .select('chain_id', 'subchain_id')
            .unique()
    )
    assembly_chain = (
            _find_covering_assemblies(assembly_subchain)
            .join(
                assembly_subchain,
                on='assembly_id',
            )
            .join(
                chain_subchain,
                on='subchain_id',
            )
            .select(
                'assembly_id',
                'chain_id',
            )
            .unique()
    )
    return assembly_chain

def _find_covering_assemblies(assembly_subchain):
    # Sometimes, multimeric structures specify separate biological assemblies 
    # for the whole multimer and the individual monomers.  I want to minimize 
    # the number of redundant chains to deal with, so to handle cases like 
    # these, I choose the minimum number of assemblies necessary to include 
    # every subchain.  This is called the set-cover problem.  In general this 
    # is a difficult problem to solve, but all the examples in the PDB are 
    # small enough to solve quickly.

    assembly_i = (
            assembly_subchain
            .select('assembly_id')
            .unique()
            .sort('assembly_id')
            .select(
                pl.int_range(pl.len()).alias('assembly_i'),
                pl.col('assembly_id'),
            )
    )

    subchain_i = (
            assembly_subchain
            .select('subchain_id')
            .unique()
            .sort('subchain_id')
            .select(
                pl.int_range(pl.len()).alias('subchain_i'),
                pl.col('subchain_id'),
            )
    )

    i = (
            assembly_subchain
            .join(assembly_i, on='assembly_id')
            .join(subchain_i, on='subchain_id')
            .select('assembly_i', 'subchain_i')
    )

    # *A* is a matrix that specifies which assemblies contain which subchains.  
    # The structure of the matrix is as follows:
    #
    # - Each column corresponds to an assembly
    # - Each row corresponds to a subchain
    # - Each value is 1 if the assembly contains the subchain, 0 otherwise.

    A = np.zeros((len(subchain_i), len(assembly_i)))
    A[i['subchain_i'], i['assembly_i']] = 1

    res = milp(
            c=np.ones(len(assembly_i)),
            integrality=np.ones(len(assembly_i)),
            bounds=Bounds(lb=0, ub=1),
            constraints=LinearConstraint(A, lb=1),
    )

    assert res.success

    covering_assembly = (
            assembly_i
            .join(
                pl.DataFrame({
                    'assembly_i': np.arange(len(assembly_i)),
                    'select': res.x.astype(int),
                }),
                on='assembly_i',
            )
            .filter(
                pl.col('select') != 0
            )
            .select('assembly_id')
    )

    return covering_assembly

def _find_chain_entity_pairs(id_map):
    return (
            id_map
            .select('chain_id', 'entity_id')
            .unique()
    )

def _find_polymers(entity_poly):
    return (
            entity_poly
            .rename({
                'pdbx_seq_one_letter_code_can': 'sequence',
            })
            .with_columns(
                pl.col('sequence').str.replace_all('\n', ''),
            )
    )

def _find_nonpolymers(entity_nonpoly):
    return None if (df := entity_nonpoly).is_empty() else df
