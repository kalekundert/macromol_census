import duckdb
import polars as pl
from contextlib import contextmanager

# The naming conventions used to refer to different parts of a PDB entry are 
# confusing and inconsistent [1].  The tables in this database follow the 
# structure → model → chain → subchain → residue hierarchy used by `gemmi`.  
# This convention seems reasonably well thought-out, and it's convenient to use 
# the same names as the parsing library we're using.
#
# [1]: https://gemmi.readthedocs.io/en/latest/mol.html#pdbx-mmcif-format

def open_db(path, read_only=False):
    return duckdb.connect(path, read_only=read_only)

def init_db(db):

    # Structures:
    db.execute('''\
            DROP TYPE IF EXISTS EXPTL_METHOD;
            CREATE TYPE EXPTL_METHOD AS ENUM (
                'ELECTRON CRYSTALLOGRAPHY',
                'ELECTRON MICROSCOPY',
                'EPR',
                'FIBER DIFFRACTION',
                'FLUORESCENCE TRANSFER',
                'INFRARED SPECTROSCOPY',
                'NEUTRON DIFFRACTION',
                'POWDER DIFFRACTION',
                'SOLID-STATE NMR',
                'SOLUTION NMR',
                'SOLUTION SCATTERING',
                'THEORETICAL MODEL',
                'X-RAY DIFFRACTION'
            );

            CREATE SEQUENCE IF NOT EXISTS structure_id;
            CREATE TABLE IF NOT EXISTS structure (
                id INT DEFAULT nextval('structure_id') PRIMARY KEY,
                pdb_id STRING NOT NULL UNIQUE,
                exptl_methods EXPTL_METHOD[],
                deposit_date DATE,
                full_atom BOOLEAN NOT NULL,
                rank INT
            );

            CREATE TABLE IF NOT EXISTS structure_blacklist (
                struct_id INT NOT NULL,
                FOREIGN KEY (struct_id) REFERENCES structure(id)
            );
    ''')

    # Clusters:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS cluster_id;
            CREATE TABLE IF NOT EXISTS cluster (
                id INT DEFAULT nextval('cluster_id') PRIMARY KEY,
                namespace STRING NOT NULL,
                name STRING NOT NULL,
                UNIQUE (name, namespace)
            );
    ''')

    # Models:
    db.execute('''\
            -- This table doesn't contain every model in the structure, only
            -- those that are consistent with the rest of the relationships
            -- stored in this database.  (There are a small number of 
            -- structures with different sets of subchains and/or different
            -- entity/subchain relationships in different models.  Fully 
            -- accounting for this would make the database much more complex,
            -- for little benefit, so instead we just keep track of which 
            -- models are actually described by the database.)

            CREATE SEQUENCE IF NOT EXISTS model_id;
            CREATE TABLE IF NOT EXISTS model (
                id INT DEFAULT nextval('model_id') PRIMARY KEY,
                struct_id INT NOT NULL,
                pdb_id STRING NOT NULL,
                FOREIGN KEY (struct_id) REFERENCES structure(id)
            );
    ''')

    # Chains:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS chain_id;
            CREATE TABLE IF NOT EXISTS chain (
                id INT DEFAULT nextval('chain_id') PRIMARY KEY,
                struct_id INT NOT NULL,
                pdb_id STRING NOT NULL,
                FOREIGN KEY(struct_id) REFERENCES structure(id)
            );
    ''')

    # Entities:
    db.execute('''\
            DROP TYPE IF EXISTS ENTITY_TYPE;
            -- According to the mmCIF/PDBx dictionary, 'macrolide' is another
            -- valid entity type.  However, as of 2024/02/14, there are no 
            -- such entities in the PDB.
            CREATE TYPE ENTITY_TYPE AS ENUM (
                'polymer',
                'non-polymer',
                'branched',
                'water'
            );

            DROP TYPE IF EXISTS POLYMER_TYPE;
            CREATE TYPE POLYMER_TYPE AS ENUM (
                'cyclic-pseudo-peptide',
                'other',
                'peptide nucleic acid',
                'polydeoxyribonucleotide',
                'polydeoxyribonucleotide/polyribonucleotide hybrid',
                'polypeptide(D)',
                'polypeptide(L)',
                'polyribonucleotide'
            );

            DROP TYPE IF EXISTS BRANCHED_TYPE;
            CREATE TYPE BRANCHED_TYPE AS ENUM (
                'oligosaccharide'
            );

            CREATE SEQUENCE IF NOT EXISTS entity_id;
            CREATE TABLE IF NOT EXISTS entity (
                id INT DEFAULT nextval('entity_id') PRIMARY KEY,
                struct_id INT NOT NULL,
                pdb_id STRING NOT NULL,
                type ENTITY_TYPE NOT NULL,
                formula_weight_Da FLOAT,
                FOREIGN KEY(struct_id) REFERENCES structure(id)
            );

            CREATE TABLE IF NOT EXISTS entity_polymer (
                entity_id INT NOT NULL,
                type POLYMER_TYPE,
                sequence STRING,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_branched (
                entity_id INT NOT NULL,
                type BRANCHED_TYPE,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_branched_bond (
                entity_id INT NOT NULL,
                pdb_seq_id_1 STRING NOT NULL,
                pdb_comp_id_1 STRING NOT NULL,
                pdb_atom_id_1 STRING NOT NULL,
                pdb_seq_id_2 STRING NOT NULL,
                pdb_comp_id_2 STRING NOT NULL,
                pdb_atom_id_2 STRING NOT NULL,
                bond_order STRING NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_monomer (
                entity_id INT NOT NULL,
                pdb_comp_id STRING NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_ignore (
                entity_id INT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_cluster (
                entity_id INT NOT NULL,
                cluster_id INT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entity(id),
                FOREIGN KEY(cluster_id) REFERENCES cluster(id)
            );
    ''')

    # Components:
    db.execute('''\
            CREATE TABLE IF NOT EXISTS component (
                pdb_id STRING PRIMARY KEY,
                inchi STRING,
                inchi_key STRING
            );
    ''')

    # Subchains:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS subchain_id;
            CREATE TABLE IF NOT EXISTS subchain (
                id INT DEFAULT nextval('subchain_id') PRIMARY KEY,
                chain_id INT NOT NULL,
                entity_id INT NOT NULL,
                pdb_id STRING NOT NULL,
                FOREIGN KEY(chain_id) REFERENCES chain(id),
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );
    ''');

    # Assemblies:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS assembly_id;
            CREATE TABLE IF NOT EXISTS assembly (
                id INT DEFAULT nextval('assembly_id') PRIMARY KEY,
                struct_id INT NOT NULL,
                pdb_id STRING NOT NULL,
                type STRING,
                polymer_count INT,
                FOREIGN KEY(struct_id) REFERENCES structure(id)
            );

            CREATE TABLE IF NOT EXISTS assembly_subchain (
                assembly_id INT NOT NULL,
                subchain_id INT NOT NULL,
                FOREIGN KEY(assembly_id) REFERENCES assembly(id),
                FOREIGN KEY(subchain_id) REFERENCES subchain(id)
            );

            CREATE TABLE IF NOT EXISTS assembly_rank (
                assembly_id INT NOT NULL,
                rank INT NOT NULL,
                FOREIGN KEY(assembly_id) REFERENCES assembly(id)
            )
    ''')

    # Quality:
    db.execute('''\
            DROP TYPE IF EXISTS MMCIF_DICT;
            CREATE TYPE MMCIF_DICT AS ENUM (
                -- These are the names of the mmCIF dictionaries that quality 
                -- data can come from.  "vrpt" stands for "validation report".
                'mmcif_pdbx',
                'mmcif_pdbx_vrpt'
            );

            CREATE TABLE IF NOT EXISTS quality_xtal (
                struct_id INT NOT NULL,
                source MMCIF_DICT NOT NULL,
                resolution_A REAL,
                r_work REAL,
                r_free REAL,
                CHECK (resolution_A > 0),
                CHECK (r_free > 0),
                CHECK (r_work > 0),
                FOREIGN KEY (struct_id) REFERENCES structure(id)
            );

            CREATE TABLE IF NOT EXISTS quality_nmr (
                struct_id INT NOT NULL,
                source MMCIF_DICT NOT NULL,
                num_dist_restraints INT,
                CHECK (num_dist_restraints > 0),
                FOREIGN KEY (struct_id) REFERENCES structure(id)
            );

            CREATE TABLE IF NOT EXISTS quality_nmr_representative (
                model_id INT NOT NULL,
                source MMCIF_DICT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS quality_em (
                struct_id INT NOT NULL,
                source MMCIF_DICT NOT NULL,
                resolution_A REAL,
                q_score REAL,
                CHECK (resolution_A > 0),
                CHECK (q_score >= -1 AND q_score <= 1),
                FOREIGN KEY (struct_id) REFERENCES structure(id)
            );

            CREATE TABLE IF NOT EXISTS quality_clashscore (
                struct_id INT NOT NULL,
                source MMCIF_DICT NOT NULL,
                clashscore REAL,
                CHECK (clashscore >= 0),
                FOREIGN KEY (struct_id) REFERENCES structure(id)
            );
    ''')

    # Redundancy:
    db.execute('''\
            CREATE TABLE IF NOT EXISTS nonredundant (
                subchain_id INT,
                FOREIGN KEY(subchain_id) REFERENCES subchain(id)
            );

            CREATE TABLE IF NOT EXISTS nonredundant_pair (
                subchain_id_1 INT,
                subchain_id_2 INT,
                FOREIGN KEY(subchain_id_1) REFERENCES subchain(id),
                FOREIGN KEY(subchain_id_2) REFERENCES subchain(id)
            );
    ''')

    db.commit()

@contextmanager
def transaction(db):
    db.execute('BEGIN TRANSACTION')
    try:
        yield
    except:
        db.execute('ROLLBACK')
        raise
    else:
        db.execute('COMMIT')


def insert_structure(
        db,
        pdb_id,
        *,
        exptl_methods,
        deposit_date,
        full_atom,
        models=None,
        assemblies,
        assembly_subchains,
        subchains,
        entities,
        polymer_entities=None,
        branched_entities=None,
        branched_entity_bonds=None,
        monomer_entities=None,
        xtal_quality=None,
        nmr_representative=None,
        em_quality=None,
):
    """
    Insert the given structure into the given database.

    The main role of this function is to translate PDB id numbers to database 
    primary key numbers.  The data frames provided to this function describe 
    all the relationships between the models, assemblies, chains, subchains, 
    and entities in the structure in terms of the id numbers used by the PDB.  
    This function works out how to express all the same relationships using 
    globally unique keys.

    This function should be used within a transaction, since the database could 
    end up in a corrupt state if a structure is only partially ingested.  
    However, responsibility for transaction handling is left to the caller.
    """

    ## Rename id columns:

    # Switch to the column naming convention where `*_id` refers to an SQL 
    # primary key and `pdb_*_id` refers to the ids used in the mmCIF file (and 
    # other PDB-associated resources).

    def label_pdb_ids(df, cols):
        if df is None: return None
        return df.sort(cols).rename({x: f'pdb_{x}' for x in cols})

    models = label_pdb_ids(models, ['id'])
    assemblies = label_pdb_ids(assemblies, ['id'])
    assembly_subchains = label_pdb_ids(
            assembly_subchains, ['assembly_id', 'subchain_id']
    )
    subchains = label_pdb_ids(subchains, ['id', 'chain_id', 'entity_id'])
    entities = label_pdb_ids(entities, ['id'])
    polymer_entities = label_pdb_ids(polymer_entities, ['entity_id'])
    branched_entities = label_pdb_ids(branched_entities, ['entity_id'])
    branched_entity_bonds = label_pdb_ids(branched_entity_bonds, ['entity_id'])
    monomer_entities = label_pdb_ids(monomer_entities, ['entity_id'])

    ## Process and sanity-check inputs:

    # Sorting the ids isn't necessary, but it makes testing easier.  I assume 
    # the runtime cost is negligible, but I haven't benchmarked it.  Note that 
    # the `label_pdb_ids()` function above also sorts by id.

    chains = (
            subchains
            .select(pdb_id=pl.col('pdb_chain_id').unique().sort())
    )

    assert not assemblies['pdb_id'].is_duplicated().any()
    assert (
            set(assemblies['pdb_id']) == 
            set(assembly_subchains['pdb_assembly_id'])
    )
    assert not subchains['pdb_id'].is_duplicated().any()
    assert (
            set(subchains['pdb_id']) == 
            set(assembly_subchains['pdb_subchain_id'])
    )

    # Some structures have entities that aren't in any biological assembly.  
    # I've found one case where this was an error (3km0) and one where it 
    # wasn't (3ttm).  I decided to address this by loosening this check, since 
    # it doesn't really hurt to have "extra" entities in the database, and 
    # having this information will let me to a more thorough review of 
    # structures like this.
    assert (
            set(entities['pdb_id']) >=
            set(subchains['pdb_entity_id'])
    )

    def check_entity_ids(types, df):
        left_ids = set(
                entities
                .filter(pl.col('type').is_in(types))
                .get_column('pdb_id')
        )
        right_ids = set() if df is None else set(df['pdb_entity_id'])
        assert left_ids == right_ids, (types, left_ids, right_ids)

    check_entity_ids(['polymer'], polymer_entities)
    check_entity_ids(['branched'], branched_entities)
    check_entity_ids(['branched'], branched_entity_bonds)
    check_entity_ids(['non-polymer', 'water'], monomer_entities)

    ## Insert everything into the database:

    struct_id = _insert_structure(
            db, pdb_id,
            exptl_methods=exptl_methods,
            deposit_date=deposit_date,
            full_atom=full_atom,
    )

    if models is not None:
        model_ids = _insert_models(db, struct_id, models)

    assembly_ids = _insert_assemblies(db, struct_id, assemblies)
    chain_ids = _insert_chains(db, struct_id, chains)
    entity_ids = _insert_entities(db, struct_id, entities)

    subchains = (
            subchains
            .join(chain_ids, on='pdb_chain_id')
            .join(entity_ids, on='pdb_entity_id')
    )
    subchain_ids = _insert_subchains(db, subchains)

    assembly_subchains = (
            assembly_subchains
            .join(assembly_ids, on='pdb_assembly_id')
            .join(subchain_ids, on='pdb_subchain_id')
    )
    _insert_assembly_subchains(db, assembly_subchains)

    if polymer_entities is not None:
        polymer_entities = (
                polymer_entities
                .join(entity_ids, on='pdb_entity_id')
        )
        _insert_polymer_entities(db, polymer_entities)

    if branched_entities is not None:
        branched_entities = (
                branched_entities
                .join(entity_ids, on='pdb_entity_id')
        )
        branched_entity_bonds = (
                branched_entity_bonds
                .join(entity_ids, on='pdb_entity_id')
        )
        _insert_branched_entities(db, branched_entities, branched_entity_bonds)

    if monomer_entities is not None:
        monomer_entities = (
                monomer_entities
                .join(entity_ids, on='pdb_entity_id')
        )
        _insert_monomer_entities(db, monomer_entities)

    if xtal_quality is not None:
        _insert_xtal_quality(db, struct_id, xtal_quality)

    if nmr_representative is not None:
        assert models is not None
        model_id = (
                model_ids
                .filter(pl.col('pdb_model_id') == nmr_representative)
                .select('model_id')
                .item()
        )
        _insert_nmr_representative(db, model_id)

    if em_quality is not None:
        _insert_em_quality(db, struct_id, em_quality)

    return struct_id

def _insert_structure(db, pdb_id, *, exptl_methods, deposit_date, full_atom):
    cur = db.execute('''\
            INSERT INTO structure (
                pdb_id,
                exptl_methods,
                deposit_date,
                full_atom
            )
            VALUES (?, ?, ?, ?)
            RETURNING id''',
            (pdb_id, exptl_methods, deposit_date, full_atom),
    )
    struct_id, = cur.fetchone()
    return struct_id

def _insert_models(db, struct_id, models):
    return _insert_pdb_ids(db, 'model', struct_id, models)

def _insert_assemblies(db, struct_id, assemblies):
    return db.execute('''\
            INSERT INTO assembly (struct_id, pdb_id, type, polymer_count)
            SELECT ?, pdb_id, type, polymer_count FROM assemblies
            RETURNING id AS assembly_id, pdb_id AS pdb_assembly_id''',
            [struct_id],
    ).pl()

def _insert_assembly_subchains(db, assembly_subchains):
    db.execute('''\
            INSERT INTO assembly_subchain (assembly_id, subchain_id)
            SELECT assembly_id, subchain_id FROM assembly_subchains
    ''')

def _insert_chains(db, struct_id, chains):
    return _insert_pdb_ids(db, 'chain', struct_id, chains)

def _insert_subchains(db, subchains):
    return db.execute('''\
            INSERT INTO subchain (chain_id, entity_id, pdb_id)
            SELECT chain_id, entity_id, pdb_id from subchains
            RETURNING id AS subchain_id, pdb_id AS pdb_subchain_id
    ''').pl()

def _insert_entities(db, struct_id, entities):
    return db.execute('''\
            INSERT INTO entity (struct_id, pdb_id, type, formula_weight_Da)
            SELECT ?, pdb_id, type, formula_weight_Da from entities
            RETURNING id AS entity_id, pdb_id AS pdb_entity_id
    ''', [struct_id]).pl()

def _insert_polymer_entities(db, polymers):
    db.execute('''\
            INSERT INTO entity_polymer (entity_id, type, sequence)
            SELECT entity_id, type, sequence from polymers
    ''')

def _insert_branched_entities(db, branched, branched_bonds):
    db.execute('''\
            INSERT INTO entity_branched (entity_id, type)
            SELECT entity_id, type FROM branched;

            INSERT INTO entity_branched_bond (
                entity_id,
                pdb_seq_id_1, pdb_comp_id_1, pdb_atom_id_1,
                pdb_seq_id_2, pdb_comp_id_2, pdb_atom_id_2,
                bond_order
            )
            SELECT 
                entity_id,
                seq_id_1, comp_id_1, atom_id_1,
                seq_id_2, comp_id_2, atom_id_2,
                bond_order
            FROM branched_bonds;
    ''')

def _insert_monomer_entities(db, monomers):
    db.execute('''\
            INSERT INTO entity_monomer (entity_id, pdb_comp_id)
            SELECT entity_id, comp_id FROM monomers
    ''')

def _insert_xtal_quality(db, struct_id, quality_df):
    db.execute('''\
            INSERT INTO quality_xtal (
                struct_id,
                source,
                resolution_A,
                r_work,
                r_free
            )
            SELECT
                ?,
                'mmcif_pdbx',
                resolution_A,
                r_work,
                r_free
            FROM quality_df
    ''', [struct_id])

def _insert_nmr_representative(db, model_id):
    db.execute('''\
            INSERT INTO quality_nmr_representative (model_id, source)
            VALUES (?, 'mmcif_pdbx')
    ''', [model_id])

def _insert_em_quality(db, struct_id, quality_df):
    db.execute('''\
            INSERT INTO quality_em (
                struct_id,
                source,
                resolution_A
            )
            SELECT
                ?,
                'mmcif_pdbx',
                resolution_A
            FROM quality_df
    ''', [struct_id])

def _insert_pdb_ids(db, table, struct_id, pdb_ids):
    return db.execute(f'''\
            INSERT INTO {table} (struct_id, pdb_id)
            SELECT ?, pdb_id FROM pdb_ids
            RETURNING id AS {table}_id, pdb_id AS pdb_{table}_id''',
            [struct_id],
    ).pl()

def update_structure_ranks(db, ranks):
    db.sql('''\
            UPDATE structure
            SET rank = ranks.rank
            FROM ranks
            WHERE structure.id = ranks.struct_id
    ''')

def insert_blacklisted_structures(db, blacklist):
    db.execute('''\
            INSERT INTO structure_blacklist (struct_id)
            SELECT structure.id
            FROM blacklist
            JOIN structure USING (pdb_id)
    ''')

def insert_assembly_ranks(db, ranks):
    """
    Arguments:
        ranks:
            A dataframe with the following columns:

            - ``assembly_id``: References to rows in the *assembly* table.
            - ``rank``: Ranks are relative to other assemblies within the same 
              structure, so it's ok for different structures to reuse the same 
              ranks.  Assemblies not assigned a rank (i.e. not present in this 
              dataframe) are considered unsuitable to include to the dataset.
    """
    db.execute('''\
            INSERT INTO assembly_rank (assembly_id, rank)
            SELECT assembly_id, rank FROM ranks
    ''')

def insert_nmr_quality(db, struct_id, *, source, num_dist_restraints=None):
    db.execute('''\
            INSERT INTO quality_nmr (struct_id, source, num_dist_restraints)
            VALUES (?, ?, ?)
    ''', [struct_id, source, num_dist_restraints])

def insert_em_quality(db, struct_id, *, source, resolution_A=None, q_score=None):
    db.execute('''\
            INSERT INTO quality_em (struct_id, source, resolution_A, q_score)
            VALUES (?, ?, ?, ?)
    ''', (struct_id, source, resolution_A, q_score))

def insert_clashscore(db, struct_id, *, source, clashscore):
    db.execute('''\
            INSERT INTO quality_clashscore (struct_id, source, clashscore)
            VALUES (?, ?, ?)
    ''', (struct_id, source, clashscore))

def insert_nonspecific_ligands(db, ignore):
    db.sql('''\
            INSERT INTO entity_ignore (entity_id)
            SELECT entity_monomer.entity_id
            FROM ignore
            JOIN entity_monomer USING (pdb_comp_id)
    ''')

def insert_entity_clusters(db, clusters, namespace):
    """
    Arguments:
        clusters:
            A dataframe with columns *entity_id* and *cluster_id*.  For former 
            must reference a row in the *entity* table.
    """

    # Ignore singleton clusters.
    clusters = (
            clusters
            .rename({'cluster_id': 'name'})
            .cast({'name': str})
            .filter(
                pl.len().over('name') > 1
            )
    )
    cluster_names = (
            clusters
            .unique('name', maintain_order=True)
    )
    cluster_ids = db.sql('''\
            INSERT INTO cluster (namespace, name)
            SELECT ?, name FROM cluster_names
            RETURNING id AS cluster_id, name
    ''', params=[namespace]).pl()

    cluster_edges = (
            clusters
            .join(cluster_ids, on='name')
    )
    db.sql('''\
            INSERT INTO entity_cluster (entity_id, cluster_id)
            SELECT entity_id, cluster_id FROM cluster_edges
    ''')

def insert_chemical_components(db, components):
    db.sql('''\
            INSERT INTO component (pdb_id, inchi, inchi_key)
            SELECT id, inchi, inchi_key
            FROM components
    ''')

def create_structure_indices(db):
    db.execute('''\
            DROP INDEX IF EXISTS structure_pdb_id;
            CREATE UNIQUE INDEX structure_pdb_id ON structure (pdb_id);
    ''')


def select_structures(db):
    return db.execute('SELECT * FROM structure').pl()

def select_structure_id(db, pdb_id):
    cur = db.execute('SELECT id FROM structure WHERE pdb_id = ?', (pdb_id,))
    return cur.fetchone()[0]

def select_blacklisted_structures(db):
    return db.execute('SELECT * FROM structure_blacklist').pl()

def select_models(db):
    return db.execute('SELECT * FROM model').pl()

def select_clusters(db):
    return db.execute('SELECT * FROM cluster').pl()

def select_assemblies(db):
    return db.execute('SELECT * FROM assembly').pl()

def select_assembly_subchains(db):
    return db.execute('SELECT * FROM assembly_subchain').pl()

def select_assembly_ranks(db):
    return db.execute('SELECT * FROM assembly_rank').pl()

def select_chains(db):
    return db.execute('SELECT * FROM chain').pl()

def select_subchains(db):
    return db.execute('SELECT * FROM subchain').pl()

def select_entities(db):
    return db.execute('SELECT * FROM entity').pl()

def select_entity_clusters(db):
    return db.execute('SELECT * FROM entity_cluster').pl()

def select_polymer_entities(db):
    return db.execute('SELECT * FROM entity_polymer').pl()

def select_branched_entities(db):
    return db.execute('SELECT * FROM entity_branched').pl()

def select_branched_entity_bonds(db):
    return db.execute('SELECT * FROM entity_branched_bond').pl()

def select_monomer_entities(db):
    return db.execute('SELECT * FROM entity_monomer').pl()

def select_ignored_entities(db):
    return db.execute('SELECT * FROM entity_ignore').pl()

def select_chemical_components(db):
    return db.execute('SELECT * FROM component').pl()

def select_xtal_quality(db):
    return db.execute('SELECT * FROM quality_xtal').pl()

def select_nmr_quality(db):
    return db.execute('SELECT * FROM quality_nmr').pl()

def select_nmr_representatives(db):
    return db.execute('SELECT * FROM quality_nmr_representative').pl()

def select_em_quality(db):
    return db.execute('SELECT * FROM quality_em').pl()

def select_clashscores(db):
    return db.execute('SELECT * FROM quality_clashscore').pl()

def select_nonredundant_subchains(db):
    return db.execute('SELECT * FROM nonredundant').pl()

def select_nonredundant_subchain_pairs(db):
    return db.execute('SELECT * FROM nonredundant_pair').pl()
