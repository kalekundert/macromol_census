import macromol_census as mmc
import polars as pl
import polars.testing

def test_metadata():
    db = mmc.open_db(':memory:')
    mmc.init_atoms(db)

    # Plausible metadata values, with an assortment of different data types.
    meta_in = {
        'pisces_max_resolution': 3.0,
        'pisces_max_identity': 70.0,
        'view_radius': 15,
        'min_atoms_per_view': 50,
        'interpro_url': 'https://www.example.com/{pdb_id}',
    }

    mmc.insert_metadata(db, meta_in)
    db.commit()

    meta_out = mmc.load_metadata(db)

    assert meta_out == meta_in

def test_atoms_structure():
    db = mmc.open_db(':memory:')
    mmc.init_atoms(db)

    assert mmc.load_pdb_ids(db) == []

    atoms = pl.DataFrame([
        dict(atom='N', chem_comp='LYS'),
        dict(atom='CA', chem_comp='LYS'),
    ])
    struct_1 = mmc.Structure(
            pdb_id='0aaa',
            atoms=atoms,
            interpro_available=True,
    )
    struct_2 = mmc.Structure(
            pdb_id='0bbb',
            atoms=atoms,
            interpro_available=False,
    )
    mmc.insert_structure(db, struct_1)
    mmc.insert_structure(db, struct_2)
    db.commit()

    assert mmc.load_pdb_ids(db, True) == ['0aaa']
    assert mmc.load_pdb_ids(db, False) == ['0aaa', '0bbb']

    struct_out = mmc.load_structure(db, '0aaa')

    assert struct_out.pdb_id == '0aaa'
    pl.testing.assert_frame_equal(struct_out.atoms, atoms)
    assert struct_out.interpro_available == True

def test_atoms_homology():
    EntryType = mmc.InterProEntryType

    db = mmc.open_db(':memory:')
    mmc.init_atoms(db)

    mmc.insert_structure(db, mmc.Structure('0aaa', pl.DataFrame(), True))
    mmc.insert_structure(db, mmc.Structure('0aab', pl.DataFrame(), True))
    mmc.insert_structure(db, mmc.Structure('0aac', pl.DataFrame(), True))
    mmc.insert_structure(db, mmc.Structure('0aad', pl.DataFrame(), True))

    entry_1 = mmc.InterProEntry('IPR000001', EntryType.DOMAIN)

    mmc.insert_interpro_homology_edge(db, '0aaa', entry_1)
    mmc.insert_interpro_homology_edge(db, '0aab', entry_1)

    mmc.insert_mmseqs2_homology_edge(db, '0aac', 1)
    mmc.insert_mmseqs2_homology_edge(db, '0aad', 1)
    db.commit()

    g = mmc.load_homology_graph(db)

    assert g.has_node(('pdb', '0aaa'))
    assert g.has_node(('pdb', '0aab'))
    assert g.has_node(('pdb', '0aac'))
    assert g.has_node(('pdb', '0aad'))
    assert g.has_node(('interpro', 'IPR000001'))
    assert g.has_node(('mmseqs2', 1))

    assert g.has_edge(('pdb', '0aaa'), ('interpro', 'IPR000001'))
    assert g.has_edge(('pdb', '0aab'), ('interpro', 'IPR000001'))
    assert g.has_edge(('pdb', '0aac'), ('mmseqs2', 1))
    assert g.has_edge(('pdb', '0aad'), ('mmseqs2', 1))

    assert g.nodes['interpro', 'IPR000001']['type'] == EntryType.DOMAIN

