import os
from tempfile import NamedTemporaryFile

import pytest

import LiSE
from LiSE.allegedb import query

query.QueryEngine.path = os.path.dirname(LiSE.__file__)


@pytest.fixture(scope="function")
def tmpdbfile():
    f = NamedTemporaryFile(suffix=".tmp.db", delete=False)
    f.close()
    yield f.name
    os.remove(f.name)
