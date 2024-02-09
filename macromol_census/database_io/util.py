import io
import polars as pl

from more_itertools import one

class NotFound(Exception):
    pass

def _adapt_dataframe(df):
    out = io.BytesIO()
    df.write_parquet(out)
    return out.getvalue()

def _convert_dataframe(bytes):
    in_ = io.BytesIO(bytes)
    df = pl.read_parquet(in_)
    return df

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

