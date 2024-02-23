"""\
Usage:
    ingest_models <in:cif-dir> <out:db-path>

Arguments:
    <dir>
        The path to a directory containing 
"""

import numpy as np
import polars as pl

from .working_db import (
        open_db, init_db, transaction,
        insert_model, select_models, create_model_indices,
)
from .error import IngestError, add_path_to_ingest_error
from gemmi.cif import read as read_cif
from scipy.optimize import milp, Bounds, LinearConstraint
from itertools import chain
from more_itertools import one
from datetime import date

def main():
    import docopt
    from pathlib import Path
    from tqdm import tqdm

    args = docopt.docopt(__doc__)
    cif_dir = Path(args['<in:cif-dir>'])
    db = open_db(args['<out:db-path>'])

    init_db(db)
    cif_paths = find_uningested_paths(
            db,
            cif_paths=tqdm(
                cif_dir.glob('**/*.cif*'),
                desc='find paths to ingest',
            ),
            pdb_id_from_path=lambda p: p.name.split('.')[0],
    )
    ingest_models(db, tqdm(cif_paths, desc='ingest models'))

def find_uningested_paths(db, cif_paths, pdb_id_from_path):

    def safe_pdb_id_from_path(path):
        pdb_id = pdb_id_from_path(path)
        assert len(pdb_id) == 4
        return pdb_id

    already_ingested = set(select_models(db)['pdb_id'].unique())
    return [
            p for p in cif_paths
            if safe_pdb_id_from_path(p) not in already_ingested
    ]

def ingest_models(db, cif_paths):
    # Multiprocessing makes this go faster, but causes weird incompatibilities 
    # with tidyexc.  This is ultimately a bug in tidyexc, and I want to fix it 
    # eventually, but for now I'm just going to return to the non-parallel 
    # algorithm.

    # from multiprocessing import get_context
    # 
    # with get_context("spawn").Pool() as pool:
    #     for kwargs in pool.imap_unordered(
    #             _get_insert_model_kwargs,
    #             cif_paths,
    #             chunksize=10,
    #     ):
    #         with transaction(db):
    #             insert_model(db, **kwargs)

    for cif_path in cif_paths:
         kwargs = _get_insert_model_kwargs(cif_path)
         with transaction(db):
             insert_model(db, **kwargs)

    create_model_indices(db)

def _get_insert_model_kwargs(cif_path):
    with add_path_to_ingest_error(cif_path):
        cif = read_cif(str(cif_path)).sole_block()
        pdb_id = cif.name.lower()

        atom_site = _extract_dataframe(
                cif, 'atom_site',
                required_cols=[
                    'auth_asym_id',
                    'label_asym_id',
                    'label_entity_id',
                    'label_seq_id',
                ],
                optional_cols=[
                    'pdbx_PDB_model_num',
                ],
        )
        struct_assembly_gen = _extract_dataframe(
                cif, 'pdbx_struct_assembly_gen',
                required_cols=[
                    'assembly_id',
                    'asym_id_list',
                ],
        )

        id_map = _make_chain_subchain_entity_id_map(atom_site)
        assembly_chain_pairs = _find_covering_assembly_chain_pairs(
                struct_assembly_gen,
                id_map,
        )
        chain_entity_pairs = _find_chain_entity_pairs(id_map)

        return dict(
                pdb_id=pdb_id,

                exptl_methods=_extract_exptl_methods(cif),
                deposit_date=_extract_deposit_date(cif),
                full_atom=_is_full_atom(atom_site),

                quality_xtal=_extract_quality_xtal(cif),
                quality_nmr=_extract_quality_nmr(cif),
                quality_em=_extract_quality_em(cif),

                assembly_chain_pairs=assembly_chain_pairs,
                chain_entity_pairs=chain_entity_pairs,

                polymers=_extract_polymers(cif),
                nonpolymers=_extract_nonpolymers(cif),
        )

def _extract_dataframe(cif, key_prefix, *, required_cols=None, optional_cols=None):
    # Gemmi automatically interprets `?` and `.`, but this leads to a few 
    # problems.  First is that it makes column dtypes dependent on the data; if 
    # a column doesn't have any non-null values, polars won't know that it 
    # should be a string.  Second is that gemmi distinguishes between `?` 
    # (null) and `.` (false).  This is a particularly unhelpful distinction 
    # when the column in question is supposed to contain float data, because 
    # the latter then becomes 0 rather than null.
    #
    # To avoid these problems, we explicitly specify a schema where each column 
    # is a string.  Doing this happens to convert any booleans present in the 
    # data to null, thereby solving both of the above problems at once.

    loop = cif.get_mmcif_category(f'_{key_prefix}.')
    df = pl.DataFrame(loop, {k: str for k in loop})

    expected_cols = list(chain(
        required_cols or [],
        optional_cols or [],
    ))

    if df.is_empty():
        schema = {col: str for col in expected_cols}
        return pl.DataFrame([], schema)

    if required_cols:
        missing_cols = [x for x in required_cols if x not in df.columns]
        if missing_cols:
            err = IngestError(
                    category=key_prefix,
                    missing_cols=missing_cols,
            )
            err.brief = "missing required column(s)"
            err.info += "category: _{category}.*"
            err.blame += "missing column(s): {missing_cols}"
            raise err

    if optional_cols:
        df = df.with_columns([
            pl.lit(None, dtype=str).alias(col)
            for col in optional_cols
            if col not in df.columns
        ])

    return (
            df
            .select(*expected_cols)
            .filter(~pl.all_horizontal(pl.all().is_null()))
    )

def _extract_exptl_methods(cif):
    exptl = _extract_dataframe(cif, 'exptl', required_cols=['method'])
    return list(exptl['method'])

def _extract_deposit_date(cif):
    table = cif.get_mmcif_category('_pdbx_database_status.')
    ymd = one(table['recvd_initial_deposition_date'])
    return date.fromisoformat(ymd)

def _extract_quality_xtal(cif):
    return (
            _extract_dataframe(
                cif, 'refine',
                optional_cols=[
                    'ls_number_reflns_obs',
                    'ls_d_res_high',
                    'ls_R_factor_R_free',
                    'ls_R_factor_R_work',
                ],
            )
            .select(
                resolution_A=pl.col('ls_d_res_high').cast(float),
                r_free=pl.col('ls_R_factor_R_free').cast(float),
                r_work=pl.col('ls_R_factor_R_work').cast(float),
                num_reflections=pl.col('ls_number_reflns_obs').cast(float),
            )
    )

def _extract_quality_nmr(cif):
    return (
            _extract_dataframe(
                cif, 'pdbx_nmr_representative',
                optional_cols=['conformer_id'],
            )
            .select('conformer_id')
    )

def _extract_quality_em(cif):
    df = (
            _extract_dataframe(
                cif, 'em_3d_reconstruction',
                optional_cols=['resolution'],
            )
            .select(
                resolution_A=pl.col('resolution').cast(float)
            )
    )

    if df.is_empty():
        return df

    # Don't call `min()` unless there are already rows in the dataframe, 
    # otherwise a row of nulls will be added.
    return df.min()

def _extract_polymers(cif):
    return (
            _extract_dataframe(
                cif, 'entity_poly',
                required_cols=[
                    'entity_id',
                    'type',
                    'pdbx_seq_one_letter_code_can',
                ],
            )
            .rename({
                'pdbx_seq_one_letter_code_can': 'sequence',
            })
            .select(
                pl.col('entity_id'),
                pl.col('type'),
                pl.col('sequence').str.replace_all('\n', ''),
            )
    )

def _extract_nonpolymers(cif):
    entity_nonpoly = _extract_dataframe(
            cif, 'pdbx_entity_nonpoly',
            required_cols=['entity_id', 'comp_id'],
    )
    return None if entity_nonpoly.is_empty() else entity_nonpoly

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

def _is_full_atom(atom_site):
    return (
            atom_site
            .lazy()
            .group_by('label_asym_id', 'label_seq_id')
            .len()
            .select((pl.col('len') > 1).any())
            .collect()
            .item()
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

