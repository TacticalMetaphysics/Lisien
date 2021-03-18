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
    assert eng._things_cache.keyframe['physical', 'kobold'][
        eng.branch][eng.turn][eng.tick]
    assert 'kobold' in eng.character['physical'].thing
    assert (0, 0) in eng.character['physical'].place
    assert (0, 1) in eng.character['physical'].portal[0, 0]
    eng.close()


def test_multi_keyframe(tempdir):
    eng = Engine(tempdir)
    inittest(eng, kobold_pos=(9, 9))
    eng.snap_keyframe()
    tick0 = eng.tick
    eng.turn = 1
    eng.character['physical'].thing['kobold']['location'] = (3, 3)
    eng.snap_keyframe()
    tick1 = eng.tick
    eng.close()
    eng = Engine(tempdir)
    assert eng._things_cache.keyframe['physical', 'kobold']['trunk'][0][tick0]\
           != eng._things_cache.keyframe['physical', 'kobold']['trunk'][1][tick1]
