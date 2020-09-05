import os

import pytest

from LiSE.engine import Engine
from LiSE.examples.kobold import inittest


@pytest.fixture
def cleanup_test_keyframe_load_init():
    yield
    os.remove('test_keyframe_load_init.db')


def test_keyframe_load_init(cleanup_test_keyframe_load_init):
    """Can load a keyframe at start of branch, including locations"""
    eng = Engine('test_keyframe_load_init.db')
    inittest(eng)
    eng.branch = 'new'
    eng.snap_keyframe()
    eng.close()
    eng = Engine('test_keyframe_load_init.db')
    assert eng._things_cache.keyframe['physical', 'kobold'][
        eng.branch][eng.turn][eng.tick]
