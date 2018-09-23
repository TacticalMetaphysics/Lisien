from LiSE import Engine
import pytest
import os


@pytest.fixture(scope='function')
def engy():
    codefiles = ('trigger.py', 'prereq.py', 'action.py', 'method.py', 'function.py')
    for file in codefiles:
        if os.path.exists(file):
            os.remove(file)
    with Engine(":memory:") as eng:
        yield eng
    for file in codefiles:
        os.remove(file)