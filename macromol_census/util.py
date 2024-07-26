import polars as pl

from .error import IngestError
from itertools import chain

MANUAL_CORRECTIONS = {}

def read_cif(path):
    from gemmi.cif import read
    return read(str(path)).sole_block()

def extract_dataframe(
        cif,
        key_prefix,
        *,
        required_cols=None,
        optional_cols=None,
):
    # Gemmi automatically interprets `?` and `.`, but this leads to a few 
    # problems.  First is that it makes column dtypes dependent on the data; if 
    # a column doesn't have any non-null values, polars won't know that it 
    # should be a string.  Second is that gemmi distinguishes between `?` 
    # (null) and `.` (false).  This is a particularly unhelpful distinction 
    # when the column in question is supposed to contain float data, because 
    # the latter then becomes 0 rather than null.

    pdb_id = cif.name.lower()

    try:
        df = MANUAL_CORRECTIONS[pdb_id, key_prefix]
    except KeyError:
        loop = {
                k: [v if isinstance(v, str) else None for v in vs]
                for k, vs in cif.get_mmcif_category(f'_{key_prefix}.').items()
        }
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

class tquiet:
    """
    Mimic the `tqdm` progress bar interface, but don't actually display 
    anything.

    This class is meant to be a default argument for functions with long 
    running loops.  This allows the main function to remain in complete control 
    of terminal output.
    """

    def __init__(self, iterable=None, total=None):
        self.iterable = iterable

    def __iter__(self):
        yield from self.iterable

    def update(self, n=1):
        pass

    def close(self):
        pass

    def set_description(self, desc=None, refresh=True):
        pass
