# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from LiSE.proxy import EngineProcessManager
import LiSE.allegedb.tests.test_all
import pytest
import LiSE.examples.kobold as kobold
import LiSE.examples.college as college
import os
import tempfile
from . import data


class ProxyTest(LiSE.allegedb.tests.test_all.AllegedTest):
    def setUp(self):
        self.manager = EngineProcessManager()
        self.engine = self.manager.start('sqlite:///:memory:')
        self.graphmakers = (self.engine.new_character,)
        self.tempdir = tempfile.mkdtemp(dir='.')
        self.addCleanup(self._do_cleanup)
        for f in (
                'trigger.py', 'prereq.py', 'action.py', 'function.py',
                'method.py', 'strings.json'
        ):
            if os.path.exists(f):
                os.rename(f, os.path.join(self.tempdir, f))

    def _do_cleanup(self):
        self.manager.shutdown()
        for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
        ):
            if os.path.exists(f):
                os.remove(f)
            if os.path.exists(os.path.join(self.tempdir, f)):
                os.rename(os.path.join(self.tempdir, f), f)
        os.rmdir(self.tempdir)


class ProxyGraphTest(LiSE.allegedb.tests.test_all.AbstractGraphTest, ProxyTest):
    pass


class DictStorageTest(ProxyTest, LiSE.allegedb.tests.test_all.DictStorageTest):
    pass


class ListStorageTest(ProxyTest, LiSE.allegedb.tests.test_all.ListStorageTest):
    pass


class SetStorageTest(ProxyTest, LiSE.allegedb.tests.test_all.SetStorageTest):
    pass


@pytest.fixture(scope='function', params=[
    lambda eng: kobold.inittest(eng, shrubberies=20, kobold_sprint_chance=.9),
    # college.install,
    # sickle.install
])
def hand(request, clean):
    from LiSE.handle import EngineHandle
    hand = EngineHandle((':memory:',), {'random_seed': 69105})
    with hand._real.advancing():
        request.param(hand._real)
    yield hand
    hand.close()


def test_fast_delta(hand):
    # just set a baseline for the diff
    hand.get_slow_delta()
    # there's currently no way to do fast delta past the time when
    # a character was created, due to the way keyframes work...
    # so don't test that
    tick = hand._real.tick
    ret, diff = hand.next_turn()
    slowd = hand.get_slow_delta()
    assert diff == slowd, "Fast delta differs from slow delta"
    ret, diff2 = hand.time_travel('trunk', 0, tick)
    slowd2 = hand.get_slow_delta()
    assert diff2 == slowd2, "Fast delta differs from slow delta"
    ret, diff3 = hand.time_travel('trunk', 3)
    slowd3 = hand.get_slow_delta()
    assert diff3 == slowd3, "Fast delta differs from slow delta"
    ret, diff4 = hand.time_travel('trunk', 1)
    slowd4 = hand.get_slow_delta()
    assert diff4 == slowd4, "Fast delta differs from slow delta"


def test_assignment(clean):
    from LiSE.handle import EngineHandle
    hand = EngineHandle((':memory:',), {'random_seed': 69105})
    eng = hand._real
    with eng.advancing():
        college.install(eng)
    physical_copy = hand.character_copy('physical')
    assert physical_copy == data.PHYSICAL_INITIAL_COPY
    dorm_copy = hand.character_copy('dorm0')
    assert dorm_copy == data.DORM_INITIAL_COPY
    student_initial_copy = data.STUDENT_INITIAL_COPY
    student_initial_copy['roommate'] = eng.character['dorm0room0student1']
    student_initial_copy['room'] = eng.character['physical'].place['dorm0room0']
    assert hand.character_copy('dorm0room0student0') == student_initial_copy
    hand.close()


def test_serialize_deleted(clean):
    from LiSE import Engine
    eng = Engine(':memory:', random_seed=69105)
    with eng.advancing():
        college.install(eng)
    d0r0s0 = eng.character['dorm0room0student0']
    roommate = d0r0s0.stat['roommate']
    del eng.character[roommate.name]
    assert not roommate
    with pytest.raises(KeyError):
        eng.character[roommate.name]
    assert d0r0s0.stat['roommate'] == roommate
    assert eng.unpack(eng.pack(d0r0s0.stat['roommate'])) == roommate


def test_manip_deleted(clean):
    from LiSE import Engine
    eng = Engine(':memory:', random_seed=69105)
    phys = eng.new_character('physical')
    phys.stat['aoeu'] = True
    phys.add_node(0)
    phys.add_node(1)
    phys.node[1]['aoeu'] = True
    del phys.node[1]
    phys.add_node(1)
    assert 'aoeu' not in phys.node[1]
    phys.add_edge(0, 1)
    phys.adj[0][1]['aoeu'] = True
    del phys.adj[0][1]
    phys.add_edge(0, 1)
    assert 'aoeu' not in phys.adj[0][1]
    del eng.character['physical']
    assert not phys
    phys = eng.new_character('physical')
    assert 'aoeu' not in phys.stat
    assert 0 not in phys
    assert 1 not in phys
    assert 0 not in phys.adj
    assert 1 not in phys.adj
