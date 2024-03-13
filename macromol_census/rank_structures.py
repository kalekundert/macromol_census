"""\
Rank each structure by accuracy.

Usage:
    mmc_rank_structures <in:db>

Arguments:
    <in:db>
        A database created by `mmc_init` and populated by both 
        `mmc_ingest_structures` and `mmc_ingest_validation`.
"""

import polars as pl
from .database_io import open_db, update_structure_ranks

def main():
    import docopt

    args = docopt.docopt(__doc__)
    db = open_db(args['<in:db>'])

    ranks = rank_structures(db)
    update_structure_ranks(db, ranks)

def rank_structures(db):
    quality_metrics = db.sql('''\
            SELECT 
                structure.id AS struct_id,
                coalesce(
                    min(quality_xtal.resolution_A),
                    min(quality_em.resolution_A),
                ) AS resolution_A,
                min(quality_clashscore.clashscore) AS clashscore,
                max(quality_nmr.num_dist_restraints) AS nmr_restraints,
                min(quality_xtal.r_free) AS xtal_r_free,
                max(quality_em.q_score) AS em_q_score,
                max(structure.deposit_date) AS deposit_date,
                max(structure.pdb_id) AS pdb_id
            FROM structure
            LEFT JOIN quality_xtal ON structure.id = quality_xtal.struct_id
            LEFT JOIN quality_nmr ON structure.id = quality_nmr.struct_id
            LEFT JOIN quality_em ON structure.id = quality_em.struct_id
            LEFT JOIN quality_clashscore ON structure.id = quality_clashscore.struct_id
            GROUP BY structure.id
    ''').pl()

    # Bin the real-valued scores so that other metrics can be used to 
    # discriminate between similar structures.

    def bin(col, bin_size):
        return (pl.col(col) / bin_size).round().cast(int)

    sort_by = {
            'resolution_bin': False,  # False: lower is better
            'clashscore_bin': False,
            'nmr_restraints': True,   # True: higher is better
            'xtal_r_free': False,
            'em_q_score': True,
            'deposit_date': True,
            'pdb_id': True,           # ensure a deterministic ordering
    }

    return (
            quality_metrics
            .with_columns(
                resolution_bin=(
                    pl.when(pl.col('resolution_A') <= 4)
                    .then(bin('resolution_A', 0.1))
                    .otherwise(None)
                ),
                clashscore_bin=bin('clashscore', 0.2),
            )
            .sort(
                *sort_by.keys(),
                descending=list(sort_by.values()),
                nulls_last=True,
            )
            .with_row_index('rank', 1)
            .select('struct_id', 'rank')
    )

