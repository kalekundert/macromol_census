import duckdb
import polars as pl

from .util import NotFound, _dataclass_row_factory, _scalar_row_factory
from dataclasses import dataclass, asdict

@dataclass(frozen=True)
class QualityXtal:
    resolution_A: float             # RESOLUTION
    reflections_per_atom: float     # REFPATM  
    r_free: int                     # RFFIN
    r_work: int                     # RFIN

def open_db(path):
    return duckdb.connect(path)

def init_db(db):

    # Models:
    db.execute('''\
            CREATE SEQUENCE IF NOT EXISTS model_id;
            CREATE TABLE IF NOT EXISTS model (
                id INT DEFAULT nextval('model_id') PRIMARY KEY,
                pdb_id STRING
            );

            CREATE TABLE IF NOT EXISTS model_quality_xtal (
                model_id INT NOT NULL,
                resolution_A REAL,
                reflections_per_atom REAL,
                r_free REAL,
                r_work REAL,
                FOREIGN KEY (model_id) REFERENCES model(id)
            );

            CREATE TABLE IF NOT EXISTS model_blacklist (
                model_id INT NOT NULL,
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
                cluster STRING,
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
                cluster STRING,
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


def insert_metadata(db, param, value):
    db.execute('''\
            INSERT INTO metadata (param, value)
            VALUES (?, ?)''',
            (param, value),
    )
    db.commit()

def insert_model(
        db,
        pdb_id,
        *,
        quality=None,
        assembly_chain_pairs,
        chain_entity_pairs,
        polymers=None,
        nonpolymers=None,
):
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

    db.execute('BEGIN TRANSACTION')

    try:
        model_id = _insert_model(db, pdb_id)

        if quality is not None:
            _insert_model_quality(db, model_id, quality)

        # Sorting the ids isn't really necessary, but it makes testing easier.  
        # I don't think the runtime cost will be noticeable, but I haven't 
        # benchmarked it yet.

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

    except:
        db.execute('ROLLBACK')
        raise

    else:
        db.execute('COMMIT')

def _insert_model(db, pdb_id):
    cur = db.execute('''\
            INSERT INTO model (pdb_id)
            VALUES (?)
            RETURNING id''',
            (pdb_id,),
    )
    model_id, = cur.fetchone()
    return model_id

def _insert_model_quality(db, model_id, quality):
    match quality:

        case QualityXtal():
            db.execute('''\
                    INSERT INTO model_quality_xtal (
                        model_id,
                        resolution_A,
                        reflections_per_atom,
                        r_free,
                        r_work
                    )
                    VALUES (
                        $model_id,
                        $resolution_A,
                        $reflections_per_atom,
                        $r_free,
                        $r_work
                    )''',
                    dict(model_id=model_id) | asdict(quality),
            )

        case _:
            raise TypeError(f"can't interpret {quality} as a model quality description")

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


def create_model_indices():
    pass


def select_models(db):
    return db.execute(f'SELECT * FROM model').pl()

def select_model_qualities_xtal(db):
    return db.execute(f'SELECT * FROM model_quality_xtal').pl()

def select_assemblies(db):
    return db.execute(f'SELECT * FROM assembly').pl()

def select_chains(db):
    return db.execute(f'SELECT * FROM chain').pl()

def select_entities(db):
    return db.execute(f'SELECT * FROM entity').pl()

def select_entity_polymers(db):
    return db.execute(f'SELECT * FROM entity_polymer').pl()

def select_entity_nonpolymers(db):
    return db.execute(f'SELECT * FROM entity_nonpolymer').pl()

def select_assembly_chain_pairs(db):
    return db.execute(f'SELECT * FROM assembly_chain').pl()

def select_chain_entity_pairs(db):
    return db.execute(f'SELECT * FROM chain_entity').pl()


