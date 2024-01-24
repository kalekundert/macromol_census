import polars as pl
import re

from .error import UsageError
from pathlib import Path
from more_itertools import pairwise

def load_pisces(path: Path):

    def optional_float(x):
        return None if x == 'NA' else float(x)

    EXPECTED_COLUMNS = ['PDBchain', 'len', 'method', 'resol', 'rfac', 'freerfac']
    COLUMN_TYPES = [str, int, str, float, optional_float, optional_float]

    with open(path) as f:
        lines = f.readlines()

    header = lines[0]
    body = lines[1:]

    if (actual := header.split()) != EXPECTED_COLUMNS:
        err = UsageError(path=path, expected=EXPECTED_COLUMNS, actual=actual)
        err.brief = "can't parse PISCES file; found unexpected columns"
        err.info += "path: {path}"
        err.info += "expected columns: {expected}"
        err.blame += "actual columns: {actual}"
        raise err

    def iter_column_boundaries(header):
        for col in header.split():
            yield header.find(col)
        yield len(header)

    df = pl.DataFrame([
        {
            col: dtype(value)
            for col, dtype, value in zip(
                    EXPECTED_COLUMNS,
                    COLUMN_TYPES,
                    line.split(),
                    strict=True,
            )
        }
        for line in body
    ])

    return (df
            .with_columns(
                pdb=pl.col('PDBchain').str.slice(0, 4).str.to_lowercase(),
                chain=pl.col('PDBchain').str.slice(4, None),
            )
            .rename({
                'resol': 'resolution_A',
                'rfac': 'R_work',
                'freerfac': 'R_free',
            })
            .select([
                'pdb',
                'chain',
                'len',
                'method',
                'resolution_A',
                'R_work',
                'R_free',
            ])
    )

def parse_pisces_path(path: Path):
    """
    Attempt to extract as much metadata as possible from the name of a file 
    downloaded from the PISCES server.
    """
    i = '[0-9]+'
    f = fr'{i}\.{i}'
    pisces_pattern = fr'''
            cullpdb_
            pc(?P<max_percent_identity>{f})_
            res(?P<min_resolution_A>{f})-(?P<max_resolution_A>{f})_
            ((?P<no_breaks>noBrks)_)?
            len(?P<min_length>{i})-(?P<max_length>{i})_
            R(?P<max_r_free>{f})_
            (?P<experiments>[a-zA-Z+]+)_
            d(?P<year>\d{{4}})_(?P<month>\d{{2}})_(?P<day>\d{{2}})_
            chains(?P<num_chains>{i})
    '''
    if m := re.match(pisces_pattern, path.name, re.VERBOSE):
        return m.groupdict()
    else:
        return {}


