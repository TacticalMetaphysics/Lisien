import os
import lisien
from tempfile import NamedTemporaryFile
import pytest
from lisien.allegedb import query

query.QueryEngine.path = os.path.dirname(lisien.__file__)


@pytest.fixture(scope="function")
def tmpdbfile():
	f = NamedTemporaryFile(suffix=".tmp.db", delete=False)
	f.close()
	yield f.name
	os.remove(f.name)
