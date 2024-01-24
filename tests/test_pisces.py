import macromol_census as mmc
import polars as pl
import polars.testing
import parametrize_from_file as pff

from pathlib import Path

with_py = pff.Namespace()

@pff.parametrize(
        schema=pff.cast(expected=with_py.eval),
        indirect=['tmp_files'],
)
def test_load_pisces(tmp_files, expected):
    actual = mmc.load_pisces(tmp_files / 'pisces.txt')
    expected = pl.DataFrame(expected)

    pl.testing.assert_frame_equal(actual, expected)

@pff.parametrize(schema=[pff.cast(path=Path), pff.defaults(expected=None)])
def test_parse_pisces_path(path, expected):
    params = mmc.parse_pisces_path(path)

    if expected is None:
        # Just make sure the pattern matched, don't check the contents.
        assert params

    else:
        if 'no_breaks' not in expected:
            expected['no_breaks'] = None
        assert params == expected

