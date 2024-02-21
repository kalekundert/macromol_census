from tidyexc import Error
from contextlib import contextmanager

class UsageError(Error):
    pass

class IngestError(Error):
    pass

@contextmanager
def add_path_to_ingest_error(path):
    with IngestError.add_info('path: {path}', path=path):
        try:
            yield

        except IngestError:
            raise

        except Exception as err:
            summary = str(err).split('\n')[0]
            raise IngestError(summary) from err
