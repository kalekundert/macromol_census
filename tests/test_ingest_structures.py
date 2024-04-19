import polars as pl
import macromol_census as mmc

from pytest import approx
from pytest_unordered import unordered
from polars.testing import assert_frame_equal
from functools import partial
from pathlib import Path
from datetime import date

CIF_DIR = Path(__file__).parent / 'pdb'

assert_frame_equal = partial(assert_frame_equal, check_dtype=False)

def test_find_uningested_paths():
    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    from test_database_io import insert_1abc
    insert_1abc(db)

    uningested_paths = mmc.find_uningested_paths(
            db,
            cif_paths=['1abc', '9xyz'],
            pdb_id_from_path=lambda x: x,
    )

    assert uningested_paths == ['9xyz']

def test_ingest_mmcif_4erd():
    # 4erd is an interesting model, because it's one of the few examples in the 
    # PDB where a single chain (an RNA double helix, in this case) appears in 
    # multiple biological assemblies.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '4erd.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='4erd',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2012, month=4, day=19),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
            dict(id=2, struct_id=1, pdb_id='B'),
            dict(id=3, struct_id=1, pdb_id='C'),
            dict(id=4, struct_id=1, pdb_id='D'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='polymer',
                formula_weight_Da=approx(15785.705),
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='polymer',
                formula_weight_Da=approx(7050.227),
            ),
            dict(
                id=3,
                struct_id=1,
                pdb_id='3',
                type='non-polymer',
                formula_weight_Da=approx(39.098),
            ),
            dict(
                id=4,
                struct_id=1,
                pdb_id='4',
                type='water',
                formula_weight_Da=approx(18.015),
            ),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='MHHHHHHSIKQNCLIKIINIPQGTLKAEVVLAVRHLGYEFYCDYIDGQAMIRFQNSDEQRLAIQKLLNHNNNKLQIEIRGQICDVISTIPEDEEKNYWNYIKFKKNEFRKFFFMKKQQKKQNITQNYNK',
            ),
            dict(
                entity_id=2,
                type='polyribonucleotide',
                sequence='GGUCGACAUCUUCGGAUGGACC',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=3, pdb_comp_id='K'),
            dict(entity_id=4, pdb_comp_id='HOH'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1, pdb_id='A', entity_id=1, chain_id=1),
            dict(id=2, pdb_id='B', entity_id=1, chain_id=2),
            dict(id=3, pdb_id='C', entity_id=2, chain_id=3),
            dict(id=4, pdb_id='D', entity_id=2, chain_id=4),
            dict(id=5, pdb_id='E', entity_id=3, chain_id=1),
            dict(id=6, pdb_id='F', entity_id=4, chain_id=1),
            dict(id=7, pdb_id='G', entity_id=4, chain_id=2),
            dict(id=8, pdb_id='H', entity_id=4, chain_id=3),
            dict(id=9, pdb_id='I', entity_id=4, chain_id=4),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_defined_assembly',
                polymer_count=3,
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='author_defined_assembly',
                polymer_count=3,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=3),
            dict(assembly_id=1, subchain_id=4),
            dict(assembly_id=1, subchain_id=5),
            dict(assembly_id=1, subchain_id=6),
            dict(assembly_id=1, subchain_id=8),
            dict(assembly_id=1, subchain_id=9),
            dict(assembly_id=2, subchain_id=2),
            dict(assembly_id=2, subchain_id=3),
            dict(assembly_id=2, subchain_id=4),
            dict(assembly_id=2, subchain_id=7),
            dict(assembly_id=2, subchain_id=8),
            dict(assembly_id=2, subchain_id=9),   
    ]
    assert mmc.select_xtal_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(2.589),
                r_work=approx(0.2197),
                r_free=approx(0.2717),
            ),
    ]
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).is_empty()

def test_ingest_mmcif_2g10():
    # 2g10 has two models with inconsistent subchain assignments.  The first 
    # model has one carbon monoxide subchain (D) while the second has two (D, 
    # E).  On top of that, the biological assembly is only compatible with the 
    # first model, which is notably the model with fewer subchains.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '2g10.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='2g10',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2006, month=2, day=13),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            # Only the first model is compatible with the biological assembly.
            dict(id=1, struct_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1', type='polymer', formula_weight_Da=approx(17399.180)),
            dict(id=2, struct_id=1, pdb_id='2', type='non-polymer', formula_weight_Da=approx(96.063)),
            dict(id=3, struct_id=1, pdb_id='3', type='non-polymer', formula_weight_Da=approx(616.487)),
            dict(id=4, struct_id=1, pdb_id='4', type='non-polymer', formula_weight_Da=approx(28.010)),
            dict(id=5, struct_id=1, pdb_id='5', type='water', formula_weight_Da=approx(18.015)),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='MVLSEGEWQLVLHVWAKVEADVAGHGQDIFIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGNFGADAQGAMNKALELFRKDIAAKYKELGYQG',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=2, pdb_comp_id='SO4'),
            dict(entity_id=3, pdb_comp_id='HEM'),
            dict(entity_id=4, pdb_comp_id='CMO'),
            dict(entity_id=5, pdb_comp_id='HOH'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1, chain_id=1, entity_id=1, pdb_id='A'),
            dict(id=2, chain_id=1, entity_id=2, pdb_id='B'),
            dict(id=3, chain_id=1, entity_id=3, pdb_id='C'),
            dict(id=4, chain_id=1, entity_id=4, pdb_id='D'),
            dict(id=5, chain_id=1, entity_id=5, pdb_id='E'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_defined_assembly',
                polymer_count=1,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=3),
            dict(assembly_id=1, subchain_id=4),
            dict(assembly_id=1, subchain_id=5),
    ]
    assert mmc.select_xtal_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(1.9),
                r_work=approx(0.045),
                r_free=approx(0.049),
            ),
    ]
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).is_empty()

def test_ingest_mmcif_4b09():
    # 4b09 has subchains that aren't included in any biological assembly.  
    # Since biological assemblies are the "unit" that are ultimately reported, 
    # it doesn't make sense to include these subchains in the database.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '4b09.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='4b09',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2012, month=6, day=29),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            # Chains K and L are not included in any biological assemblies.
            dict(id=1,  struct_id=1, pdb_id='A'),
            dict(id=2,  struct_id=1, pdb_id='B'),
            dict(id=3,  struct_id=1, pdb_id='C'),
            dict(id=4,  struct_id=1, pdb_id='D'),
            dict(id=5,  struct_id=1, pdb_id='E'),
            dict(id=6,  struct_id=1, pdb_id='F'),
            dict(id=7,  struct_id=1, pdb_id='G'),
            dict(id=8,  struct_id=1, pdb_id='H'),
            dict(id=9,  struct_id=1, pdb_id='I'),
            dict(id=10, struct_id=1, pdb_id='J'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='polymer',
                formula_weight_Da=approx(27931.006),
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='non-polymer',
                formula_weight_Da=approx(2044.535),
            ),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='MTELPIDENTPRILIVEDEPKLGQLLIDYLRAASYAPTLISHGDQVLPYVRQTPPDLILLDLMLPGTDGLMLCREIRRFSDIPIVMVTAKIEEIDRLLGLEIGADDYICKPYSPREVVARVKTILRRCKPQRELQQQDAESPLIIDEGRFQASWRGKMLDLTPAEFRLLKTLSHEPGKVFSREQLLNHLYDDYRVVTDRTIDSHIKNLRRKLESLDAEQSFIRAVYGVGYRWEADACRIV',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=2, pdb_comp_id='TBR'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            # Subchains K, L, and U are not included in any biological 
            # assemblies.
            dict(id=1,  pdb_id='A', entity_id=1, chain_id=1),
            dict(id=2,  pdb_id='B', entity_id=1, chain_id=2),
            dict(id=3,  pdb_id='C', entity_id=1, chain_id=3),
            dict(id=4,  pdb_id='D', entity_id=1, chain_id=4),
            dict(id=5,  pdb_id='E', entity_id=1, chain_id=5),
            dict(id=6,  pdb_id='F', entity_id=1, chain_id=6),
            dict(id=7,  pdb_id='G', entity_id=1, chain_id=7),
            dict(id=8,  pdb_id='H', entity_id=1, chain_id=8),
            dict(id=9,  pdb_id='I', entity_id=1, chain_id=9),
            dict(id=10, pdb_id='J', entity_id=1, chain_id=10),
            dict(id=11, pdb_id='M', entity_id=2, chain_id=1),
            dict(id=12, pdb_id='N', entity_id=2, chain_id=2),
            dict(id=13, pdb_id='O', entity_id=2, chain_id=3),
            dict(id=14, pdb_id='P', entity_id=2, chain_id=4),
            dict(id=15, pdb_id='Q', entity_id=2, chain_id=5),
            dict(id=16, pdb_id='R', entity_id=2, chain_id=6),
            dict(id=17, pdb_id='S', entity_id=2, chain_id=7),
            dict(id=18, pdb_id='T', entity_id=2, chain_id=9),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_and_software_defined_assembly',
                polymer_count=2, 
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='author_and_software_defined_assembly',
                polymer_count=2, 
            ),
            dict(
                id=3,
                struct_id=1,
                pdb_id='3',
                type='author_and_software_defined_assembly',
                polymer_count=2, 
            ),
            dict(
                id=4,
                struct_id=1,
                pdb_id='4',
                type='author_and_software_defined_assembly',
                polymer_count=2, 
            ),
            dict(
                id=5,
                struct_id=1,
                pdb_id='5',
                type='author_and_software_defined_assembly',
                polymer_count=2, 
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == unordered([
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=11),
            dict(assembly_id=1, subchain_id=12),

            dict(assembly_id=2, subchain_id=3),
            dict(assembly_id=2, subchain_id=4),
            dict(assembly_id=2, subchain_id=13),
            dict(assembly_id=2, subchain_id=14),

            dict(assembly_id=3, subchain_id=5),
            dict(assembly_id=3, subchain_id=6),
            dict(assembly_id=3, subchain_id=15),
            dict(assembly_id=3, subchain_id=16),

            dict(assembly_id=4, subchain_id=7),
            dict(assembly_id=4, subchain_id=8),
            dict(assembly_id=4, subchain_id=17),

            dict(assembly_id=5, subchain_id=9),
            dict(assembly_id=5, subchain_id=10),
            dict(assembly_id=5, subchain_id=18),
    ])
    assert mmc.select_xtal_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(3.3),
                r_work=approx(0.2252),
                r_free=approx(0.2448),
            ),
    ]
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).is_empty()

def test_ingest_mmcif_146d():
    # 146d has two "branched" entities; a di- and tri-saccharide.  These 
    # entities are rare, but are considered neither polymers nor non-polymers, 
    # and have to be handled specially.  

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '146d.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='146d',
                exptl_methods=['SOLUTION NMR'],
                deposit_date=date(year=1993, month=11, day=9),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            # Note that both models have all the same 
            # subchain/chain/entity/assembly relationships.
            dict(id=1, struct_id=1, pdb_id='1'),
            dict(id=2, struct_id=1, pdb_id='2'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
            dict(id=2, struct_id=1, pdb_id='B'),
            dict(id=3, struct_id=1, pdb_id='C'),
            dict(id=4, struct_id=1, pdb_id='D'),
            dict(id=5, struct_id=1, pdb_id='E'),
            dict(id=6, struct_id=1, pdb_id='F'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='polymer',
                formula_weight_Da=approx(1809.217),
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='branched',
                formula_weight_Da=approx(278.299),
            ),
            dict(
                id=3,
                struct_id=1,
                pdb_id='3',
                type='branched',
                formula_weight_Da=approx(422.468),
            ),
            dict(
                id=4,
                struct_id=1,
                pdb_id='4',
                type='non-polymer',
                formula_weight_Da=approx(24.305),
            ),
            dict(
                id=5,
                struct_id=1,
                pdb_id='5',
                type='non-polymer',
                formula_weight_Da=approx(388.411),
            ),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polydeoxyribonucleotide',
                sequence='TCGCGA',
            ),
    ]
    assert mmc.select_branched_entities(db).to_dicts() == [
            dict(entity_id=2, type='oligosaccharide'),
            dict(entity_id=3, type='oligosaccharide'),
    ]
    assert mmc.select_branched_entity_bonds(db).to_dicts() == [
            dict(
                entity_id=2,
                pdb_seq_id_1='2',
                pdb_comp_id_1='DDA',
                pdb_atom_id_1='C1',
                pdb_seq_id_2='1',
                pdb_comp_id_2='DDA',
                pdb_atom_id_2='O3',
                bond_order='sing',
            ),
            dict(
                entity_id=3,
                pdb_seq_id_1='2',
                pdb_comp_id_1='DDL',
                pdb_atom_id_1='C1',
                pdb_seq_id_2='1',
                pdb_comp_id_2='DDA',
                pdb_atom_id_2='O3',
                bond_order='sing',
            ),
            dict(
                entity_id=3,
                pdb_seq_id_1='3',
                pdb_comp_id_1='MDA',
                pdb_atom_id_1='C1',
                pdb_seq_id_2='2',
                pdb_comp_id_2='DDL',
                pdb_atom_id_2='O3',
                bond_order='sing',
            ),
    ]
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=4, pdb_comp_id='MG'),
            dict(entity_id=5, pdb_comp_id='CRH'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1, entity_id=1, chain_id=1, pdb_id='A'),
            dict(id=2, entity_id=1, chain_id=2, pdb_id='B'),
            dict(id=3, entity_id=2, chain_id=3, pdb_id='C'),
            dict(id=4, entity_id=3, chain_id=4, pdb_id='D'),
            dict(id=5, entity_id=2, chain_id=5, pdb_id='E'),
            dict(id=6, entity_id=3, chain_id=6, pdb_id='F'),
            dict(id=7, entity_id=4, chain_id=1, pdb_id='G'),
            dict(id=8, entity_id=5, chain_id=1, pdb_id='H'),
            dict(id=9, entity_id=5, chain_id=2, pdb_id='I'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_defined_assembly',
                polymer_count=2,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=3),
            dict(assembly_id=1, subchain_id=4),
            dict(assembly_id=1, subchain_id=5),
            dict(assembly_id=1, subchain_id=6),
            dict(assembly_id=1, subchain_id=7),
            dict(assembly_id=1, subchain_id=8),
            dict(assembly_id=1, subchain_id=9),
    ]
    assert mmc.select_xtal_quality(db).is_empty()
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).is_empty()

def test_ingest_mmcif_6wiv():
    # 6wiv causes problems because it specifies `_refine.ls_d_res_high` as `.`.  
    # Somehow this ends up getting interpreted as 0, which give incorrect 
    # results when filtering for the lowest resolution structures.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '6wiv.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='6wiv',
                exptl_methods=['ELECTRON MICROSCOPY'],
                deposit_date=date(year=2020, month=4, day=10),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
            dict(id=2, struct_id=1, pdb_id='B'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='polymer',
                formula_weight_Da=approx(91585.867),
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='polymer',
                formula_weight_Da=approx(93278.781),
            ),
            dict(
                id=3,
                struct_id=1,
                pdb_id='3',
                type='non-polymer',
                formula_weight_Da=approx(221.208),
            ),
            dict(
                id=4,
                struct_id=1,
                pdb_id='4',
                type='non-polymer',
                formula_weight_Da=approx(40.078),
            ),
            dict(
                id=5,
                struct_id=1,
                pdb_id='5',
                type='non-polymer',
                formula_weight_Da=approx(766.039),
            ),
            dict(
                id=6,
                struct_id=1,
                pdb_id='6',
                type='non-polymer',
                formula_weight_Da=approx(386.654),
            ),
            dict(
                id=7,
                struct_id=1,
                pdb_id='7',
                type='non-polymer',
                formula_weight_Da=approx(814.167),
            ),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='MGPGAPFARVGWPLPLLVVMAAGVAPVWASHSPHLPRPHSRVPPHPSSERRAVYIGALFPMSGGWPGGQACQPAVEMALEDVNSRRDILPDYELKLIHHDSKCDPGQATKYLYELLYNDPIKIILMPGCSSVSTLVAEAARMWNLIVLSYGSSSPALSNRQRFPTFFRTHPSATLHNPTRVKLFEKWGWKKIATIQQTTEVFTSTLDDLEERVKEAGIEITFRQSFFSDPAVPVKNLKRQDARIIVGLFYETEARKVFCEVYKERLFGKKYVWFLIGWYADNWFKIYDPSINCTVDEMTEAVEGHITTEIVMLNPANTRSISNMTSQEFVEKLTKRLKRHPEETGGFQEAPLAYDAIWALALALNKTSGGGGRSGVRLEDFNYNNQTITDQIYRAMNSSSFEGVSGHVVFDASGSRMAWTLIEQLQGGSYKKIGYYDSTKDDLSWSKTDKWIGGSPPADQTLVIKTFRFLSQKLFISVSVLSSLGIVLAVVCLSFNIYNSHVRYIQNSQPNLNNLTAVGCSLALAAVFPLGLDGYHIGRNQFPFVCQARLWLLGLGFSLGYGSMFTKIWWVHTVFTKKEEKKEWRKTLEPWKLYATVGLLVGMDVLTLAIWQIVDPLHRTIETFAKEEPKEDIDVSILPQLEHCSSRKMNTWLGIFYGYKGLLLLLGIFLAYETKSVSTEKINDHRAVGMAIYNVAVLCLITAPVTMILSSQQDAAFAFASLAIVFSSYITLVVLFVPKMRRLITRGEWQSEAQDTMKTGSSTNNNEEEKSRLLEKENRELEKIIAEKEERVSELRHQLQSRDYKDDDDK',
            ),
            dict(
                entity_id=2,
                type='polypeptide(L)',
                sequence='MASPRSSGQPGPPPPPPPPPARLLLLLLLPLLLPLAPGAWGWARGAPRPPPSSPPLSIMGLMPLTKEVAKGSIGRGVLPAVELAIEQIRNESLLRPYFLDLRLYDTECDNAKGLKAFYDAIKYGPNHLMVFGGVCPSVTSIIAESLQGWNLVQLSFAATTPVLADKKKYPYFFRTVPSDNAVNPAILKLLKHYQWKRVGTLTQDVQRFSEVRNDLTGVLYGEDIEISDTESFSNDPCTSVKKLKGNDVRIILGQFDQNMAAKVFCCAYEENMYGSKYQWIIPGWYEPSWWEQVHTEANSSRCLRKNLLAAMEGYIGVDFEPLSSKQIKTISGKTPQQYEREYNNKRSGVGPSKFHGYAYDGIWVIAKTLQRAMETLHASSRHQRIQDFNYTDHTLGRIILNAMNETNFFGVTGQVVFRNGERMGTIKFTQFQDSREVKVGEYNAVADTLEIINDTIRFQGSEPPKDKTIILEQLRKISLPLYSILSALTILGMIMASAFLFFNIKNRNQKLIKMSSPYMNNLIILGGMLSYASIFLFGLDGSFVSEKTFETLCTVRTWILTVGYTTAFGAMFAKTWRVHAIFKNVKMKKKIIKDQKLLVIVGGMLLIDLCILICWQAVDPLRRTVEKYSMEPDPAGRDISIRPLLEHCENTHMTIWLGIVYAYKGLLMLFGCFLAWETRNVSIPALNDSKYIGMSVYNVGIMCIIGAAVSFLTRDQPNVQFCIVALVIIFCSTITLCLVFVPKLITLRTNPDAATQNRRFQFTQNQKKEDSKTSTSVTSVNQASTSRLEGLQSENHRLRMKITELDKDLEEVTMQLQDTDYKDDDDK',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=3, pdb_comp_id='NAG'),
            dict(entity_id=4, pdb_comp_id='CA'),
            dict(entity_id=5, pdb_comp_id='U3G'),
            dict(entity_id=6, pdb_comp_id='CLR'),
            dict(entity_id=7, pdb_comp_id='U3D'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id= 1, chain_id=1, entity_id=1, pdb_id='A'),
            dict(id= 2, chain_id=2, entity_id=2, pdb_id='B'),
            dict(id= 3, chain_id=1, entity_id=3, pdb_id='C'),
            dict(id= 4, chain_id=1, entity_id=3, pdb_id='D'),
            dict(id= 5, chain_id=1, entity_id=3, pdb_id='E'),
            dict(id= 6, chain_id=1, entity_id=4, pdb_id='F'),
            dict(id= 7, chain_id=1, entity_id=5, pdb_id='G'),
            dict(id= 8, chain_id=1, entity_id=6, pdb_id='H'),
            dict(id= 9, chain_id=1, entity_id=6, pdb_id='I'),
            dict(id=10, chain_id=1, entity_id=6, pdb_id='J'),
            dict(id=11, chain_id=1, entity_id=6, pdb_id='K'),
            dict(id=12, chain_id=2, entity_id=3, pdb_id='L'),
            dict(id=13, chain_id=2, entity_id=7, pdb_id='M'),
            dict(id=14, chain_id=2, entity_id=6, pdb_id='N'),
            dict(id=15, chain_id=2, entity_id=6, pdb_id='O'),
            dict(id=16, chain_id=2, entity_id=6, pdb_id='P'),
            dict(id=17, chain_id=2, entity_id=6, pdb_id='Q'),
            dict(id=18, chain_id=2, entity_id=6, pdb_id='R'),
            dict(id=19, chain_id=2, entity_id=6, pdb_id='S'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_defined_assembly',
                polymer_count=2,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=3),
            dict(assembly_id=1, subchain_id=4),
            dict(assembly_id=1, subchain_id=5),
            dict(assembly_id=1, subchain_id=6),
            dict(assembly_id=1, subchain_id=7),
            dict(assembly_id=1, subchain_id=8),
            dict(assembly_id=1, subchain_id=9),
            dict(assembly_id=1, subchain_id=10),
            dict(assembly_id=1, subchain_id=11),
            dict(assembly_id=1, subchain_id=12),
            dict(assembly_id=1, subchain_id=13),
            dict(assembly_id=1, subchain_id=14),
            dict(assembly_id=1, subchain_id=15),
            dict(assembly_id=1, subchain_id=16),
            dict(assembly_id=1, subchain_id=17),
            dict(assembly_id=1, subchain_id=18),
            dict(assembly_id=1, subchain_id=19),
    ]
    assert mmc.select_xtal_quality(db).is_empty()
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(3.3),
                q_score=None,
            ),
    ]

def test_ingest_mmcif_2iy3():
    # 2iy3 is not a full-atom structure; it contains protein and RNA, but only 
    # specifies a single coordinate for each residue.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '2iy3.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='2iy3',
                exptl_methods=['ELECTRON MICROSCOPY'],
                deposit_date=date(year=2006, month=7, day=12),
                full_atom=False,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
            dict(id=2, struct_id=1, pdb_id='B'),
            dict(id=3, struct_id=1, pdb_id='C'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='1', type='polymer', formula_weight_Da=approx(48274.094)),
            dict(id=2, struct_id=1, pdb_id='2', type='polymer', formula_weight_Da=approx(35547.090)),
            dict(id=3, struct_id=1, pdb_id='3', type='polymer', formula_weight_Da=approx(1380.632)),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='MFQQLSARLQEAIGRLRGRGRITEEDLKATLREIRRALMDADVNLEVARDFVERVREEALGKQVLESLTPAEVILATVYEALKEALGGEARLPVLKDRNLWFLVGLQGSGKTTTAAKLALYYKGKGRRPLLVAADTQRPAAREQLRLLGEKVGVPVLEVMDGESPESIRRRVEEKARLEARDLILVDTAGRLQIDEPLMGELARLKEVLGPDEVLLVLDAMTGQEALSVARAFDEKVGVTGLVLTKLDGDARGGAALSARHVTGKPIYFAGVSEKPEGLEPFYPERLAGRILGMGDIESILEKVKGLEEYDKIQKKMEDVMEGKGKLTLRDVYAQIIALRKMGPLSKVLQHIPGLGIMLPTPSEDQLKIGEEKIRRWLAALNSMTYKELENPNIIDKSRMRRIAEGSGLEVEEVRELLEWYNNMNRLLKMVK',
            ),
            dict(
                entity_id=2,
                type='polyribonucleotide',
                sequence='GGGGGCUCUGUUGGUUCUCCCGCAACGCUACUCUGUUUACCAGGUCAGGUCCGAAAGGAAGCAGCCAAGGCAGAUGACGCGUGUGCCGGGAUGUAGCUGGCAGGGCCCCC',
            ),
            dict(
                entity_id=3,
                type='polypeptide(L)',
                sequence='AALALAAAAALALAAAG',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).is_empty()
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1, chain_id=1, entity_id=1, pdb_id='A'),
            dict(id=2, chain_id=2, entity_id=2, pdb_id='B'),
            dict(id=3, chain_id=3, entity_id=3, pdb_id='C'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_and_software_defined_assembly',
                polymer_count=3,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=3),
    ]
    assert mmc.select_xtal_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(16),
                r_work=None,
                r_free=None,
            ),
    ]
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(16),
                q_score=None,
            ),
    ]

def test_ingest_mmcif_5i1r():
    # 5i1r is an NMR structure that specifies "all" as its representative model 
    # (as opposed to a model id number, which is what most structures do).

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '5i1r.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='5i1r',
                exptl_methods=['SOLUTION NMR', 'SOLUTION SCATTERING'],
                deposit_date=date(year=2016, month=2, day=5),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            # Note the alphabetical sort order.  This annoys me, but isn't 
            # really a problem.
            dict(id=1,  struct_id=1, pdb_id='1'),
            dict(id=2,  struct_id=1, pdb_id='10'),
            dict(id=3,  struct_id=1, pdb_id='11'),
            dict(id=4,  struct_id=1, pdb_id='12'),
            dict(id=5,  struct_id=1, pdb_id='13'),
            dict(id=6,  struct_id=1, pdb_id='14'),
            dict(id=7,  struct_id=1, pdb_id='15'),
            dict(id=8,  struct_id=1, pdb_id='16'),
            dict(id=9,  struct_id=1, pdb_id='17'),
            dict(id=10, struct_id=1, pdb_id='18'),
            dict(id=11, struct_id=1, pdb_id='19'),
            dict(id=12, struct_id=1, pdb_id='2'),
            dict(id=13, struct_id=1, pdb_id='20'),
            dict(id=14, struct_id=1, pdb_id='21'),
            dict(id=15, struct_id=1, pdb_id='3'),
            dict(id=16, struct_id=1, pdb_id='4'),
            dict(id=17, struct_id=1, pdb_id='5'),
            dict(id=18, struct_id=1, pdb_id='6'),
            dict(id=19, struct_id=1, pdb_id='7'),
            dict(id=20, struct_id=1, pdb_id='8'),
            dict(id=21, struct_id=1, pdb_id='9'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='polymer',
                formula_weight_Da=approx(6442.522),
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='non-polymer',
                formula_weight_Da=approx(65.409),
            ),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='MQRGNFRNQRKIVKCFNCGKEGHTARNCRAPRKKGCWKCGKEGHQMKDCTERQAN',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=2, pdb_comp_id='ZN'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1, chain_id=1, entity_id=1, pdb_id='A'),
            dict(id=2, chain_id=1, entity_id=2, pdb_id='B'),
            dict(id=3, chain_id=1, entity_id=2, pdb_id='C'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_and_software_defined_assembly',
                polymer_count=1,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=3),
    ]
    assert mmc.select_xtal_quality(db).is_empty()

    # Although the NMR representative field does have a value in this 
    # structure, it doesn't match the name of any of the actual models, so it 
    # is ignored.
    assert mmc.select_nmr_representatives(db).is_empty()

    assert mmc.select_em_quality(db).is_empty()

def test_ingest_mmcif_6igg():
    # 6igg has an assembly that uses all the subchains twice, with two 
    # different matrices.  This triggered a bug in my code where duplicate 
    # entries would appear in the assembly/subchain link table.

    db = mmc.open_db(':memory:')
    mmc.init_db(db)

    mmc.ingest_structures(db, [CIF_DIR / '6igg.cif.gz'])

    assert mmc.select_structures(db).to_dicts() == [
            dict(
                id=1,
                pdb_id='6igg',
                exptl_methods=['X-RAY DIFFRACTION'],
                deposit_date=date(year=2018, month=9, day=25),
                full_atom=True,
                rank=None,
            ),
    ]
    assert mmc.select_models(db).to_dicts() == [
            dict(id=1,  struct_id=1, pdb_id='1'),
    ]
    assert mmc.select_chains(db).to_dicts() == [
            dict(id=1, struct_id=1, pdb_id='A'),
    ]
    assert mmc.select_entities(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='polymer',
                formula_weight_Da=approx(19433.9),
            ),
            dict(
                id=2,
                struct_id=1,
                pdb_id='2',
                type='non-polymer',
                formula_weight_Da=approx(62.068),
            ),
            dict(
                id=3,
                struct_id=1,
                pdb_id='3',
                type='water',
                formula_weight_Da=approx(18.015),
            ),
    ]
    assert mmc.select_polymer_entities(db).to_dicts() == [
            dict(
                entity_id=1,
                type='polypeptide(L)',
                sequence='ISHMSINIRDPLIVSRVVGDVLDPFNRSITLKVTYGQREVTNGLDLRPSQVQNKPRVEIGGEDLRNFYTLVMVDPDVPSPSNPHLREYLHWLVTDIPATTGTTFGNEIVSYENPSPTAGIHRVVFILFRQLGRQTVYAPGWRQNFNTREFAEIYNLGLPVAAVFYNSQRES',
            ),
    ]
    assert mmc.select_branched_entities(db).is_empty()
    assert mmc.select_branched_entity_bonds(db).is_empty()
    assert mmc.select_monomer_entities(db).to_dicts() == [
            dict(entity_id=2, pdb_comp_id='EDO'),
            dict(entity_id=3, pdb_comp_id='HOH'),
    ]
    assert mmc.select_subchains(db).to_dicts() == [
            dict(id=1,  chain_id=1, entity_id=1, pdb_id='A'),
            dict(id=2,  chain_id=1, entity_id=2, pdb_id='B'),
            dict(id=3,  chain_id=1, entity_id=2, pdb_id='C'),
            dict(id=4,  chain_id=1, entity_id=2, pdb_id='D'),
            dict(id=5,  chain_id=1, entity_id=2, pdb_id='E'),
            dict(id=6,  chain_id=1, entity_id=2, pdb_id='F'),
            dict(id=7,  chain_id=1, entity_id=2, pdb_id='G'),
            dict(id=8,  chain_id=1, entity_id=2, pdb_id='H'),
            dict(id=9,  chain_id=1, entity_id=2, pdb_id='I'),
            dict(id=10, chain_id=1, entity_id=3, pdb_id='J'),
    ]
    assert mmc.select_assemblies(db).to_dicts() == [
            dict(
                id=1,
                struct_id=1,
                pdb_id='1',
                type='author_defined_assembly',
                polymer_count=2,
            ),
    ]
    assert mmc.select_assembly_subchains(db).to_dicts() == [
            dict(assembly_id=1, subchain_id=1),
            dict(assembly_id=1, subchain_id=2),
            dict(assembly_id=1, subchain_id=3),
            dict(assembly_id=1, subchain_id=4),
            dict(assembly_id=1, subchain_id=5),
            dict(assembly_id=1, subchain_id=6),
            dict(assembly_id=1, subchain_id=7),
            dict(assembly_id=1, subchain_id=8),
            dict(assembly_id=1, subchain_id=9),
            dict(assembly_id=1, subchain_id=10),
    ]
    assert mmc.select_xtal_quality(db).to_dicts() == [
            dict(
                struct_id=1,
                source='mmcif_pdbx',
                resolution_A=approx(1.00),
                r_work=approx(0.1163),
                r_free=approx(0.1356),
            ),
    ]
    assert mmc.select_nmr_representatives(db).is_empty()
    assert mmc.select_em_quality(db).is_empty()

def test_find_subchains():
    from macromol_census.ingest_structures import _find_subchains

    atom_site = pl.DataFrame([
        dict(auth_asym_id='AAA', label_asym_id='A', label_entity_id='1'),
        dict(auth_asym_id='AAA', label_asym_id='A', label_entity_id='1'),

        dict(auth_asym_id='AAA', label_asym_id='B', label_entity_id='2'),
        dict(auth_asym_id='AAA', label_asym_id='B', label_entity_id='2'),

        dict(auth_asym_id='AAA', label_asym_id='C', label_entity_id='3'),
        dict(auth_asym_id='AAA', label_asym_id='C', label_entity_id='3'),

        dict(auth_asym_id='BBB', label_asym_id='D', label_entity_id='1'),
        dict(auth_asym_id='BBB', label_asym_id='D', label_entity_id='1'),

        dict(auth_asym_id='BBB', label_asym_id='E', label_entity_id='2'),
        dict(auth_asym_id='BBB', label_asym_id='E', label_entity_id='2'),

        dict(auth_asym_id='BBB', label_asym_id='F', label_entity_id='3'),
        dict(auth_asym_id='BBB', label_asym_id='F', label_entity_id='3'),
    ])

    actual = _find_subchains(atom_site)
    expected = pl.DataFrame([
        dict(id='A', chain_id='AAA', entity_id='1'),
        dict(id='B', chain_id='AAA', entity_id='2'),
        dict(id='C', chain_id='AAA', entity_id='3'),
        dict(id='D', chain_id='BBB', entity_id='1'),
        dict(id='E', chain_id='BBB', entity_id='2'),
        dict(id='F', chain_id='BBB', entity_id='3'),
    ])

    assert_frame_equal(actual, expected, check_row_order=False)

