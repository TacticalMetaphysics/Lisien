from LiSE import Engine
import pytest
import os
import tempfile


@pytest.fixture(scope="function")
def clean():
    codefiles = ('trigger.py', 'prereq.py', 'action.py', 'method.py', 'function.py', 'strings.json')
    tempdir = tempfile.mkdtemp(dir='.')
    for file in codefiles:
        if os.path.exists(file):
            os.rename(file, os.path.join(tempdir, file))
    yield
    for file in codefiles:
        os.remove(file)
        if os.path.exists(os.path.join(tempdir, file)):
            os.rename(os.path.join(tempdir, file), file)
    os.rmdir(tempdir)


@pytest.fixture(scope='function')
def engy(clean):
    with Engine(":memory:") as eng:
        yield eng
