"""\
Usage:
    ingest_structures <in:db-path> <in:cif-dir> [--skip-huge]

Arguments:
    <in:cif-dir>
        The path to a directory containing the structures to ingest, in mmCIF 
        format.  The directory must be organized in the same way as the PDB, 
        e.g.: <in:cif-dir>/xy/9xyz.cif.gz

Options:
    --skip-huge
        Skip structures with `*.cif.gz` files that are bigger than 50 MB.  
        These structures require much more memory to process, so it can be 
        convenient to process them all at once after the bulk of the PDB has 
        been ingested (possibly on a more powerful computer).
"""

import polars as pl

from .database_io import (
        open_db, transaction,
        insert_structure, select_structures, create_structure_indices,
)
from .util import read_cif, extract_dataframe
from .error import IngestError, add_path_to_ingest_error
from more_itertools import one
from datetime import date

def main():
    import docopt
    from pathlib import Path
    from tqdm import tqdm

    args = docopt.docopt(__doc__)
    cif_dir = Path(args['<in:cif-dir>'])
    db = open_db(args['<in:db-path>'])

    cif_paths = find_uningested_paths(
            db,
            cif_paths=tqdm(
                cif_dir.glob('**/*.cif*'),
                desc='find paths to ingest',
            ),
            pdb_id_from_path=lambda p: p.name.split('.')[0],
            skip_huge=args['--skip-huge'],
    )
    ingest_structures(db, tqdm(cif_paths, desc='ingest structures'))

def find_uningested_paths(db, cif_paths, *, pdb_id_from_path, skip_huge=False):

    def safe_pdb_id_from_path(path):
        pdb_id = pdb_id_from_path(path)
        assert len(pdb_id) == 4
        return pdb_id

    already_ingested = set(select_structures(db)['pdb_id'].unique())
    return [
            p for p in cif_paths
            if (safe_pdb_id_from_path(p) not in already_ingested)
            and ((not skip_huge) or p.stat().st_size < 50_000_000)
    ]

def ingest_structures(db, cif_paths):
    # Multiprocessing makes this go faster, but causes weird incompatibilities 
    # with tidyexc.  This is ultimately a bug in tidyexc, and I want to fix it 
    # eventually, but for now I'm just going to return to the non-parallel 
    # algorithm.

    # from multiprocessing import get_context
    # 
    # with get_context("spawn").Pool() as pool:
    #     for kwargs in pool.imap_unordered(
    #             _get_insert_structure_kwargs,
    #             cif_paths,
    #             chunksize=10,
    #     ):
    #         with transaction(db):
    #             insert_structure(db, **kwargs)

    for cif_path in cif_paths:
        with add_path_to_ingest_error(cif_path):
             kwargs = _get_insert_structure_kwargs(cif_path)
             with transaction(db):
                 insert_structure(db, **kwargs)

    create_structure_indices(db)

def _get_insert_structure_kwargs(cif_path):
    cif = read_cif(cif_path)
    pdb_id = cif.name.lower()

    models, assemblies, subchains, assembly_subchains, full_atom = \
            _extract_models_subchains_assemblies(cif)

    return dict(
            pdb_id=pdb_id,
            exptl_methods=_extract_exptl_methods(cif),
            deposit_date=_extract_deposit_date(cif),
            full_atom=full_atom,

            models=models,
            assemblies=assemblies,
            subchains=subchains,
            entities=_extract_entities(cif),
            assembly_subchains=assembly_subchains,

            polymer_entities=_extract_polymer_entities(cif),
            branched_entities=_extract_branched_entities(cif),
            branched_entity_bonds=_extract_branched_entity_bonds(cif),
            monomer_entities=_extract_monomer_entities(cif),

            xtal_quality=_extract_xtal_quality(cif),
            nmr_representative=_extract_nmr_representative(cif, models),
            em_quality=_extract_em_quality(cif),
    )

def _extract_models_subchains_assemblies(cif):
    # This is a complicated function because it isn't necessarily true that 
    # each model will have the same subchain/chain/entity/assembly 
    # relationships.  So the role of this function is to find and return the 
    # relationships that describe the most models.

    atom_site = extract_dataframe(
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
    struct_assembly = extract_dataframe(
            cif, 'pdbx_struct_assembly',
            required_cols=[
                'id',
                'details',
                'oligomeric_count',
            ],
    )
    struct_assembly_gen = extract_dataframe(
            cif, 'pdbx_struct_assembly_gen',
            required_cols=[
                'assembly_id',
                'asym_id_list',
            ],
    )

    assemblies = (
            struct_assembly
            .select(
                pl.col('id'),
                pl.col('details').alias('type'),
                pl.col('oligomeric_count').cast(int).alias('polymer_count'),
            )
    )

    if struct_assembly_gen.is_empty():
        assembly_subchains = (
                atom_site
                .select(
                    pl.lit(assemblies['id'].item()).alias('assembly_id'),
                    pl.col('label_asym_id').unique(maintain_order=True).alias('subchain_id'),
                )
        )
    else:
        assert set(struct_assembly['id']) == set(struct_assembly_gen['assembly_id'])
        assembly_subchains = (
                struct_assembly_gen
                .select(
                    'assembly_id',
                    subchain_id=pl.col('asym_id_list').str.split(','),
                )
                .group_by('assembly_id', maintain_order=True)
                .agg(pl.col('subchain_id').flatten().unique(maintain_order=True))
                .explode('subchain_id')
        )

    required_subchains = set(assembly_subchains['subchain_id'])

    model_ids = {True: {}, False: {}}
    model_ids_filtered = {}
    model_attrs = {}

    for (model_id,), model_atom_site in atom_site.group_by(['pdbx_PDB_model_num']):
        all_subchains = _find_subchains(model_atom_site)
        subchains = (
                all_subchains
                .filter(
                    pl.col('id').is_in(assembly_subchains['subchain_id']),
                )
        )

        if set(subchains['id']) != required_subchains:
            continue

        exact_match = set(all_subchains['id']) == required_subchains
        full_atom = _is_full_atom(model_atom_site)
        key = frozenset(subchains.iter_rows()), full_atom

        model_ids[exact_match].setdefault(key, []).append(model_id)
        model_attrs[key] = subchains, assembly_subchains, full_atom

    # If there are models that have exactly the same subchains as the 
    # biological assemblies, prefer those.  Otherwise, consider models that 
    # have more subchains than the biological assemblies.  Note that either 
    # way, only those subchains that are actually in a biological assembly will 
    # be added to the database.

    if model_ids[True]:
        model_ids = model_ids[True]
    elif model_ids[False]:
        model_ids = model_ids[False]
    else:
        raise IngestError("no model contains every subchain in biological assembly")

    def most_common(k):
        return -len(model_ids[k])

    best_key = max(model_ids, key=most_common)

    return (
            pl.DataFrame({'id': model_ids[best_key]}),
            assemblies,
            *model_attrs[best_key],
    )

def _find_subchains(atom_site):
    return (
            atom_site
            .select(
                id='label_asym_id',
                chain_id='auth_asym_id',
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

def _extract_exptl_methods(cif):
    exptl = extract_dataframe(cif, 'exptl', required_cols=['method'])
    return list(exptl['method'])

def _extract_deposit_date(cif):
    table = cif.get_mmcif_category('_pdbx_database_status.')
    ymd = one(table['recvd_initial_deposition_date'])
    return date.fromisoformat(ymd)

def _extract_entities(cif):
    return (
            extract_dataframe(
                cif, 'entity',
                required_cols=['id'],
                optional_cols=['type', 'formula_weight'],
            )
            .rename({
                'formula_weight': 'formula_weight_Da',
            })
    )

def _extract_polymer_entities(cif):
    return (
            extract_dataframe(
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
            .with_columns(
                pl.col('sequence').str.replace_all('\n', ''),
            )
    )

def _extract_branched_entities(cif):
    entity_branch = extract_dataframe(
            cif, 'pdbx_entity_branch',
            required_cols=['entity_id', 'type'],
    )
    return None if entity_branch.is_empty() else entity_branch

def _extract_branched_entity_bonds(cif):
    entity_branch_link = (
            extract_dataframe(
                cif, 'pdbx_entity_branch_link',
                required_cols=[
                    'entity_id',
                    'entity_branch_list_num_1',
                    'comp_id_1',
                    'atom_id_1',
                    'entity_branch_list_num_2',
                    'comp_id_2',
                    'atom_id_2',
                    'value_order',
                ],
            )
            .rename({
                'entity_branch_list_num_1': 'seq_id_1',
                'entity_branch_list_num_2': 'seq_id_2',
                'value_order': 'bond_order',
            })
    )
    return None if entity_branch_link.is_empty() else entity_branch_link

def _extract_monomer_entities(cif):
    entity_monomer = extract_dataframe(
            cif, 'pdbx_entity_nonpoly',
            required_cols=['entity_id', 'comp_id'],
    )
    return None if entity_monomer.is_empty() else entity_monomer

def _extract_xtal_quality(cif):
    return (
            extract_dataframe(
                cif, 'refine',
                optional_cols=[
                    'ls_d_res_high',
                    'ls_R_factor_R_free',
                    'ls_R_factor_R_work',
                ],
            )
            .rename({
                'ls_d_res_high': 'resolution_A',
                'ls_R_factor_R_free': 'r_free',
                'ls_R_factor_R_work': 'r_work',
            })
            .select(
                pl.col('*').cast(float).replace(0, None),
            )
    )

def _extract_nmr_representative(cif, models):
    df = (
            extract_dataframe(
                cif, 'pdbx_nmr_representative',
                optional_cols=['conformer_id'],
            )
            .filter(pl.col('conformer_id').is_in(models['id']))
            .select('conformer_id')
    )
    return None if df.is_empty() else df.item()

def _extract_em_quality(cif):
    df = (
            extract_dataframe(
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


