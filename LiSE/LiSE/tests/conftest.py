from LiSE import Engine
import pytest
import os
import tempfile

codefiles = ('trigger.py', 'prereq.py', 'action.py', 'method.py', 'function.py', 'strings.json')

def preclear(tempdir):
    for file in codefiles:
        if os.path.exists(file):
            os.rename(file, os.path.join(tempdir, file))


def cleanup(tempdir):
    for file in codefiles:
        os.remove(file)
        if os.path.exists(os.path.join(tempdir, file)):
            os.rename(os.path.join(tempdir, file), file)
    os.rmdir(tempdir)


@pytest.fixture(scope='module')
def clean_module():
    tempdir = tempfile.mkdtemp(dir='.')
    preclear(tempdir)
    yield
    cleanup(tempdir)


@pytest.fixture(scope='function')
def clean():
    tempdir = tempfile.mkdtemp(dir='.')
    preclear(tempdir)
    yield
    cleanup(tempdir)


@pytest.fixture(scope='function')
def engy(clean):
    with Engine(":memory:") as eng:
        yield eng
