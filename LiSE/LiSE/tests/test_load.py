import os
import shutil

import pytest

from LiSE.engine import Engine
from LiSE.examples.kobold import inittest



def test_keyframe_load_init(tempdir):
    """Can load a keyframe at start of branch, including locations"""
    eng = Engine(tempdir)
    inittest(eng)
    eng.branch = 'new'
    eng.snap_keyframe()
    eng.close()
    eng = Engine(tempdir)
    assert 'kobold' in eng.character['physical'].thing
    assert (0, 0) in eng.character['physical'].place
    assert (0, 1) in eng.character['physical'].portal[0, 0]
    eng.close()


def test_multi_keyframe(tempdir):
    eng = Engine(tempdir)
    inittest(eng)
    eng.snap_keyframe()
    tick0 = eng.tick
    eng.turn = 1
    del eng.character['physical'].place[3, 3]
    eng.snap_keyframe()
    tick1 = eng.tick
    assert ('physical',) in eng._nodes_cache.keyframe
    assert 'trunk' in eng._nodes_cache.keyframe['physical',]
    assert 1 in eng._nodes_cache.keyframe['physical',]['trunk']
    assert tick1 in eng._nodes_cache.keyframe['physical',]['trunk'][1]
    assert (1, 1) in eng._nodes_cache.keyframe['physical',]['trunk'][1][tick1]
    assert (3, 3) not in eng._nodes_cache.keyframe['physical',]['trunk'][1][tick1]
    eng.close()
    eng = Engine(tempdir)
    assert 1 in eng._nodes_cache.keyframe['physical',]['trunk']
    assert tick1 in eng._nodes_cache.keyframe['physical',]['trunk'][1]
    eng._load_at('trunk', 0, tick0)
    assert eng._time_is_loaded('trunk', 0, tick0)
    assert eng._time_is_loaded('trunk', 0, tick0+1)
    assert eng._time_is_loaded('trunk', 1, tick1-1)
    assert eng._time_is_loaded('trunk', 1, tick1)
    assert 0 in eng._nodes_cache.keyframe['physical',]['trunk']
    assert tick0 in eng._nodes_cache.keyframe['physical',]['trunk'][0]
    assert 1 in eng._nodes_cache.keyframe['physical',]['trunk']
    assert tick1 in eng._nodes_cache.keyframe['physical',]['trunk'][1]
    assert eng._nodes_cache.keyframe['physical', ]['trunk'][0][tick0]\
           != eng._nodes_cache.keyframe['physical', ]['trunk'][1][tick1]


def test_keyframe_load_unload(tempdir):
    """Make sure all of the caches can load and unload before and after kfs"""
    with Engine(tempdir) as eng:
        eng.snap_keyframe()
        eng.turn = 1
        inittest(eng)
        eng.snap_keyframe()
        eng.turn = 2
        eng.universal['hi'] = 'hello'
        now = eng._btt()
    with Engine(tempdir) as eng:
        assert eng._time_is_loaded(*now)
        assert not eng._time_is_loaded('trunk', 0)
        eng.turn = 1
        eng.tick = 0
        assert eng._time_is_loaded('trunk', 1)
        assert eng._time_is_loaded('trunk', 1, 0)
        assert not eng._time_is_loaded('trunk', 0)
        assert eng._time_is_loaded(*now)
        eng.unload()
        assert eng._time_is_loaded('trunk', 1, 0)
        assert not eng._time_is_loaded(*now)
