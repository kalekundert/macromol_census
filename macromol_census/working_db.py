import duckdb
import polars as pl
from contextlib import contextmanager

def open_db(path):
    return duckdb.connect(path)

def init_db(db):

    # Models:
    try:
        db.execute('''\
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
        ''')
    except duckdb.CatalogException:
        pass

    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS model_id;
            CREATE TABLE IF NOT EXISTS model (
                id INT DEFAULT nextval('model_id') PRIMARY KEY,
                pdb_id STRING,
                exptl_methods EXPTL_METHOD[],
                deposit_date DATE,
                num_atoms INT,
            );

            CREATE TABLE IF NOT EXISTS model_blacklist (
                model_id INT NOT NULL,
                FOREIGN KEY (model_id) REFERENCES model(id)
            );
    ''')

    # Quality
    db.execute('''\
            CREATE TABLE IF NOT EXISTS quality_xtal (
                model_id INT NOT NULL,
                resolution_A REAL,
                num_reflections REAL,
                r_free REAL,
                r_work REAL,
                CHECK (resolution_A > 0),
                CHECK (num_reflections > 0),
                CHECK (r_free > 0),
                CHECK (r_work > 0),
                FOREIGN KEY (model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS quality_nmr (
                model_id INT NOT NULL,
                pdb_conformer_id STRING,
                num_dist_restraints INT,
                CHECK (num_dist_restraints > 0),
                FOREIGN KEY (model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS quality_em (
                model_id INT NOT NULL,
                resolution_A REAL,
                q_score REAL,
                CHECK (resolution_A > 0),
                CHECK (q_score >= -1 AND q_score <= 1),
                FOREIGN KEY (model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS quality_clashscore (
                model_id INT NOT NULL,
                clashscore REAL,
                CHECK (clashscore >= 0),
                FOREIGN KEY (model_id) REFERENCES model(id)
            );

    ''')

    # Assemblies:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS assembly_id;
            CREATE TABLE IF NOT EXISTS assembly (
                id INT DEFAULT nextval('assembly_id') PRIMARY KEY,
                model_id INT NOT NULL,
                pdb_id STRING,
                FOREIGN KEY(model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS assembly_cover (
                assembly_id INT NOT NULL,
                FOREIGN KEY(assembly_id) REFERENCES assembly(id)
            )
    ''')

    # Chains:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS chain_id;
            CREATE TABLE IF NOT EXISTS chain (
                id INT DEFAULT nextval('chain_id') PRIMARY KEY,
                model_id INT NOT NULL,
                pdb_id STRING,
                FOREIGN KEY(model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS chain_cluster (
                chain_id INT NOT NULL,
                cluster INT NOT NULL,
                FOREIGN KEY(chain_id) REFERENCES chain(id)
            )
    ''')

    # Entities:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS entity_id;
            CREATE TABLE IF NOT EXISTS entity (
                id INT DEFAULT nextval('entity_id') PRIMARY KEY,
                model_id INT NOT NULL,
                pdb_id STRING,
                FOREIGN KEY(model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS entity_polymer (
                entity_id INT NOT NULL,
                type STRING,
                sequence STRING,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_nonpolymer (
                entity_id INT NOT NULL,
                pdb_comp_id STRING,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );

            CREATE TABLE IF NOT EXISTS entity_cluster (
                entity_id INT NOT NULL,
                cluster INT NOT NULL,
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            );
    ''')

    # Redundancy:
    db.execute('''\
            CREATE TABLE IF NOT EXISTS redundancy (
                assembly_id INT,
                chain_id INT,
                redundant BOOLEAN,
                FOREIGN KEY(assembly_id) REFERENCES assembly(id),
                FOREIGN KEY(chain_id) REFERENCES chain(id)
            )
    ''')

    # Many-to-many relationships:
    db.execute('''\
            CREATE TABLE IF NOT EXISTS assembly_chain (
                assembly_id INT NOT NULL,
                chain_id INT NOT NULL,
                FOREIGN KEY(assembly_id) REFERENCES assembly(id),
                FOREIGN KEY(chain_id) REFERENCES chain(id)
            );

            CREATE TABLE IF NOT EXISTS chain_entity (
                chain_id INT NOT NULL,
                entity_id INT NOT NULL,
                FOREIGN KEY(chain_id) REFERENCES chain(id),
                FOREIGN KEY(entity_id) REFERENCES entity(id)
            )
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


def insert_model(
        db,
        pdb_id,
        *,
        exptl_methods,
        deposit_date,
        num_atoms,
        quality_xtal=None,
        quality_nmr=None,
        quality_em=None,
        assembly_chain_pairs,
        chain_entity_pairs,
        polymers=None,
        nonpolymers=None,
):
    """
    Insert the given model into the given database.

    The main role of this function is to translate PDB id numbers to database 
    primary key numbers.  The data frames provided to this function describe 
    all the relationships between the assemblies, chains, and entities in the 
    model in terms of the id numbers used in the PDB.  This function works out 
    how to express all the same relationships using globally unique keys.

    This function should be used within a transaction, since the database could 
    end up in a corrupt state if a model is only partially ingested.  However, 
    responsibility for transaction handling is left to the caller.
    """
    # Switch to the convention where `id` refers to an SQL primary key and 
    # `pdb_id` refers to the ids used in the mmCIF file (and other 
    # PDB-associated resources).

    assembly_chain_pairs = assembly_chain_pairs.rename({
        'assembly_id': 'assembly_pdb_id',
        'chain_id': 'chain_pdb_id',
    })
    chain_entity_pairs = chain_entity_pairs.rename({
        'chain_id': 'chain_pdb_id',
        'entity_id': 'entity_pdb_id',
    })

    model_id = _insert_model(
            db, pdb_id,
            exptl_methods=exptl_methods,
            deposit_date=deposit_date,
            num_atoms=num_atoms,
    )

    # Sorting the ids isn't really necessary, but it makes testing easier.  
    # I don't think the runtime cost will be noticeable, but I haven't 
    # benchmarked it.

    assembly_pdb_ids = (
            assembly_chain_pairs
            .select(pl.col('assembly_pdb_id').unique().sort())
    )
    assembly_ids = (
            _insert_assemblies(db, model_id, assembly_pdb_ids)
            .select(pl.col('*').name.prefix('assembly_'))
    )

    chain_pdb_ids = (
            assembly_chain_pairs
            .select(pl.col('chain_pdb_id').unique().sort())
    )
    chain_ids = (
            _insert_chains(db, model_id, chain_pdb_ids)
            .select(pl.col('*').name.prefix('chain_'))
    )

    entity_pdb_ids = (
            chain_entity_pairs
            .select(pl.col('entity_pdb_id').unique().sort())
    )
    entity_ids = (
            _insert_entities(db, model_id, entity_pdb_ids)
            .select(pl.col('*').name.prefix('entity_'))
    )

    if quality_xtal is not None:
        _insert_quality_xtal(db, model_id, quality_xtal)

    if quality_nmr is not None:
        _insert_quality_nmr(db, model_id, quality_nmr)

    if quality_em is not None:
        _insert_quality_em(db, model_id, quality_em)

    if polymers is not None:
        polymers = (
                polymers
                .rename({'entity_id': 'entity_pdb_id'})
                .join(entity_ids, on='entity_pdb_id')
        )
        _insert_entity_polymers(db, polymers)

    if nonpolymers is not None:
        nonpolymers = (
                nonpolymers
                .rename({'entity_id': 'entity_pdb_id'})
                .join(entity_ids, on='entity_pdb_id')
        )
        _insert_entity_nonpolymers(db, nonpolymers)

    assembly_chain_pairs = (
            assembly_chain_pairs
            .join(assembly_ids, on='assembly_pdb_id')
            .join(chain_ids, on='chain_pdb_id')
    )
    chain_entity_pairs = (
            chain_entity_pairs
            .join(chain_ids, on='chain_pdb_id')
            .join(entity_ids, on='entity_pdb_id')
    )

    _insert_assembly_chain_pairs(db, assembly_chain_pairs)
    _insert_chain_entity_pairs(db, chain_entity_pairs)

def _insert_model(db, pdb_id, *, exptl_methods, deposit_date, num_atoms):
    cur = db.execute('''\
            INSERT INTO model (pdb_id, exptl_methods, deposit_date, num_atoms)
            VALUES (?, ?, ?, ?)
            RETURNING id''',
            (pdb_id, exptl_methods, deposit_date, num_atoms),
    )
    model_id, = cur.fetchone()
    return model_id

def _insert_quality_xtal(db, model_id, quality_df):
    rows = [
        (model_id, *row)
        for row in quality_df.select(
                'resolution_A',
                'num_reflections',
                'r_free',
                'r_work',
        ).iter_rows()
    ]
    db.executemany('''\
            INSERT INTO quality_xtal (
                model_id,
                resolution_A,
                num_reflections,
                r_free,
                r_work
            )
            VALUES (?, ?, ?, ?, ?)
    ''', rows)

def _insert_quality_nmr(db, model_id, quality_df):
    rows = [
        (model_id, conf_id)
        for conf_id in quality_df['conformer_id']
    ]
    db.executemany('''\
            INSERT INTO quality_nmr (model_id, pdb_conformer_id)
            VALUES (?, ?)
    ''', rows)

def _insert_quality_em(db, model_id, quality_df):
    rows = [
        (model_id, resolution_A)
        for resolution_A in quality_df['resolution_A']
    ]
    db.executemany('''\
            INSERT INTO quality_em (model_id, resolution_A)
            VALUES (?, ?)
    ''', rows)

def _insert_assemblies(db, model_id, assembly_pdb_ids):
    return _insert_pdb_ids(db, 'assembly', model_id, assembly_pdb_ids)

def _insert_chains(db, model_id, chain_pdb_ids):
    return _insert_pdb_ids(db, 'chain', model_id, chain_pdb_ids)

def _insert_entities(db, model_id, entity_pdb_ids):
    return _insert_pdb_ids(db, 'entity', model_id, entity_pdb_ids)

def _insert_entity_polymers(db, polymers):
    db.executemany('''\
            INSERT INTO entity_polymer (entity_id, type, sequence)
            VALUES (?, ?, ?)''',
            polymers.select('entity_id', 'type', 'sequence').iter_rows(),
    )

def _insert_entity_nonpolymers(db, nonpolymers):
    db.executemany('''\
            INSERT INTO entity_nonpolymer (entity_id, pdb_comp_id)
            VALUES (?, ?)''',
            nonpolymers.select('entity_id', 'comp_id').iter_rows(),
    )

def _insert_pdb_ids(db, table, model_id, pdb_ids):
    return db.execute(f'''\
            INSERT INTO {table} (model_id, pdb_id)
            SELECT ?, * from pdb_ids
            RETURNING id, pdb_id''',
            [model_id],
    ).pl()

def _insert_assembly_chain_pairs(db, assembly_chain_pairs):
    db.executemany('''\
            INSERT INTO assembly_chain (assembly_id, chain_id)
            VALUES (?, ?)''',
            assembly_chain_pairs.select('assembly_id', 'chain_id').iter_rows(),
    )

def _insert_chain_entity_pairs(db, chain_entity_pairs):
    db.executemany('''\
            INSERT INTO chain_entity (chain_id, entity_id)
            VALUES (?, ?)''',
            chain_entity_pairs.select('chain_id', 'entity_id').iter_rows(),
    )

def insert_model_blacklist(db, blacklist):
    db.sql('''\
            INSERT INTO model_blacklist (model_id)
            SELECT model.id
            FROM blacklist
            JOIN model USING (pdb_id)
    ''')

def update_quality_nmr(db, model_id, *, num_dist_restraints=None):
    # It would be safer to insert a new row instead of updating a row that, in 
    # principle, might not exist, or might not be unique.  But I've separately 
    # measured that each NMR structure will have exactly one row in this table.
    db.execute('''\
            UPDATE quality_nmr
            SET num_dist_restraints = ?
            WHERE model_id = ?
    ''', (num_dist_restraints, model_id))

def insert_quality_em(db, model_id, *, resolution_A=None, q_score=None):
    db.execute('''\
            INSERT INTO quality_em (model_id, resolution_A, q_score)
            VALUES (?, ?, ?)
    ''', (model_id, resolution_A, q_score))

def insert_quality_clashscore(db, model_id, clashscore):
    db.execute('''\
            INSERT INTO quality_clashscore (model_id, clashscore)
            VALUES (?, ?)
    ''', (model_id, clashscore))

def insert_chain_clusters(db, clusters):
    db.sql('INSERT INTO chain_cluster SELECT chain_id, cluster FROM clusters')

def insert_entity_clusters(db, clusters):
    db.sql('INSERT INTO entity_cluster SELECT entity_id, cluster FROM clusters')


def create_model_indices(db):
    db.execute('''\
            DROP INDEX IF EXISTS model_pdb_id;
            CREATE UNIQUE INDEX model_pdb_id ON model (pdb_id);
    ''')


def select_models(db):
    return db.execute('SELECT * FROM model').pl()

def select_model_id(db, pdb_id):
    cur = db.execute('SELECT id FROM model WHERE pdb_id = ?', (pdb_id,))
    return cur.fetchone()[0]

def select_model_blacklist(db):
    return db.execute('SELECT * FROM model_blacklist').pl()

def select_qualities_xtal(db):
    return db.execute('SELECT * FROM quality_xtal').pl()

def select_qualities_nmr(db):
    return db.execute('SELECT * FROM quality_nmr').pl()

def select_qualities_em(db):
    return db.execute('SELECT * FROM quality_em').pl()

def select_qualities_clashscore(db):
    return db.execute('SELECT * FROM quality_clashscore').pl()

def select_assemblies(db):
    return db.execute('SELECT * FROM assembly').pl()

def select_chains(db):
    return db.execute('SELECT * FROM chain').pl()

def select_chain_clusters(db):
    return db.execute('SELECT * FROM chain_cluster').pl()

def select_entities(db):
    return db.execute('SELECT * FROM entity').pl()

def select_entity_polymers(db):
    return db.execute('SELECT * FROM entity_polymer').pl()

def select_entity_nonpolymers(db):
    return db.execute('SELECT * FROM entity_nonpolymer').pl()

def select_entity_clusters(db):
    return db.execute('SELECT * FROM entity_cluster').pl()

def select_assembly_chain_pairs(db):
    return db.execute('SELECT * FROM assembly_chain').pl()

def select_chain_entity_pairs(db):
    return db.execute('SELECT * FROM chain_entity').pl()


