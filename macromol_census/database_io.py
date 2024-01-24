import sqlite3
import numpy as np
import polars as pl
import networkx as nx
import io

from enum import Enum
from dataclasses import dataclass
from more_itertools import one

@dataclass
class Structure:
    pdb_id: str
    atoms: pl.DataFrame
    interpro_available: bool

class InterProEntryType(Enum):
    DOMAIN = 'domain'
    FAMILY = 'family'
    SUPERFAMILY = 'superfamily'

@dataclass
class InterProEntry:
    id: str
    type: InterProEntryType

class NotFound(Exception):
    pass

def open_db(path):
    """
    .. warning::
        It's not safe to fork the database connection object returned by this 
        function.  Thus, either avoid using the ``"fork"`` multiprocessing 
        context (e.g. with ``torch.utils.data.DataLoader``), or don't open the 
        database until already within the subprocess.

    .. warning::
        The database connection returned by this function does not have 
        autocommit behavior enabled, so the caller is responsible for 
        committing/rolling back transactions as necessary.
    """
    sqlite3.register_adapter(pl.DataFrame, _adapt_dataframe)
    sqlite3.register_converter('DATA_FRAME', _convert_dataframe)

    sqlite3.register_adapter(InterProEntryType, _adapt_interpro_entry_type)
    sqlite3.register_converter('ENTRY_TYPE', _convert_interpro_entry_type)

    db = sqlite3.connect(
            path,
            detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    db.execute('PRAGMA foreign_keys = ON')
    db.execute('PRAGMA encoding = UTF8')

    return db

def init_db(db):
    cur = db.cursor()

    cur.execute('''\
            CREATE TABLE IF NOT EXISTS metadata (
                key UNIQUE,
                value
            )
    ''')
    cur.execute('''\
            CREATE TABLE IF NOT EXISTS structures (
                pdb_id STRING PRIMARY KEY,
                atoms_parquet DATA_FRAME,
                interpro_available BOOLEAN
            )
    ''')
    cur.execute('''\
            CREATE TABLE interpro_entries (
                interpro_id TEXT PRIMARY KEY,
                type ENTRY_TYPE
            )
    ''')
    cur.execute('''\
            CREATE TABLE homology_interpro (
                pdb_id TEXT,
                interpro_id TEXT,
                FOREIGN KEY(pdb_id) REFERENCES structures(pdb_id),
                FOREIGN KEY(interpro_id) REFERENCES interpro_entries(interpro_id)
            )
    ''')
    cur.execute('''\
            CREATE TABLE homology_mmseqs2 (
                pdb_id TEXT,
                cluster_id INTEGER,
                FOREIGN KEY(pdb_id) REFERENCES structures(pdb_id)
            )
    ''')
    db.commit()


def insert_metadata(db, meta):
    db.executemany(
            'INSERT INTO metadata (key, value) VALUES (?, ?)',
            meta.items(),
    )

def insert_structure(db, structure):
    db.execute('''\
            INSERT INTO structures (pdb_id, atoms_parquet, interpro_available)
            VALUES (?, ?, ?)''',
            (structure.pdb_id, structure.atoms, structure.interpro_available),
    )

def insert_interpro_homology_edge(db, pdb_id, interpro_entry):
    db.execute('''\
            INSERT OR IGNORE INTO interpro_entries (interpro_id, type)
            VALUES (?, ?)''',
            (interpro_entry.id, interpro_entry.type),
    )
    db.execute('''\
            INSERT INTO homology_interpro (pdb_id, interpro_id)
            VALUES (?, ?)''',
            (pdb_id, interpro_entry.id),
    )

def insert_mmseqs2_homology_edge(db, pdb_id, cluster_id):
    db.execute('''\
            INSERT INTO homology_mmseqs2 (pdb_id, cluster_id)
            VALUES (?, ?)''',
            (pdb_id, cluster_id),
    )


def load_metadata(db):
    cur = db.execute('SELECT key, value FROM metadata')
    return dict(cur.fetchall())

def load_structure(db, pdb_id):
    cur = db.execute(
            'SELECT * FROM structures WHERE pdb_id=?',
            (pdb_id,),
    )
    cur.row_factory = _dataclass_row_factory(
            Structure,
            {'atoms_parquet': 'atoms'},
    )

    if struct := cur.fetchone():
        return struct
    else:
        raise NotFound(f"can't find structure with PDB id: {pdb_id}")

def load_pdb_ids(db, require_interpro=True):
    if require_interpro:
        sql = 'SELECT pdb_id FROM structures WHERE interpro_available=TRUE'
    else:
        sql = 'SELECT pdb_id FROM structures'

    cur = db.execute(sql)
    cur.row_factory = _scalar_row_factory

    return cur.fetchall()

def load_homology_graph(db):
    g = nx.Graph()

    # PDB nodes
    g.add_nodes_from(
            (('pdb', pdb_id), {})
            for pdb_id in load_pdb_ids(db)
    )

    # InterPro nodes
    cur = db.execute('SELECT interpro_id, type FROM interpro_entries')
    g.add_nodes_from(
            (('interpro', interpro_id), dict(type=interpro_type))
            for interpro_id, interpro_type in cur.fetchall()
    )
    cur = db.execute('SELECT pdb_id, interpro_id FROM homology_interpro')
    g.add_edges_from(
            (('pdb', pdb_id), ('interpro', interpro_id))
            for pdb_id, interpro_id in cur.fetchall()
    )

    # MMseqs2 nodes
    cur = db.execute('SELECT pdb_id, cluster_id FROM homology_mmseqs2')
    g.add_edges_from(
            (('pdb', pdb_id), ('mmseqs2', cluster_id))
            for pdb_id, cluster_id in cur.fetchall()
    )

    return g


def _adapt_dataframe(df):
    out = io.BytesIO()
    df.write_parquet(out)
    return out.getvalue()

def _convert_dataframe(bytes):
    in_ = io.BytesIO(bytes)
    df = pl.read_parquet(in_)
    return df

def _adapt_interpro_entry_type(entry_type):
    return entry_type.value

def _convert_interpro_entry_type(bytes):
    return InterProEntryType(bytes.decode('utf-8'))

def _dataclass_row_factory(cls, col_map={}):

    def factory(cur, row):
        row_dict = {
                col_map.get(k := col[0], k): value
                for col, value in zip(cur.description, row)
        }
        return cls(**row_dict)

    return factory

def _scalar_row_factory(cur, row):
    return one(row)

