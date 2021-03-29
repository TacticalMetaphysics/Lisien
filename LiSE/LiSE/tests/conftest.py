from LiSE import Engine
import pytest
import os
import shutil
import tempfile


@pytest.fixture(scope='function')
def tempdir():
    directory = tempfile.mkdtemp()
    yield directory
    shutil.rmtree(directory)

@pytest.fixture(scope='function')
def engy(tempdir):
    with Engine(tempdir) as eng:
        yield eng
