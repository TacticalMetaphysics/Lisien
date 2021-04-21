# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
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
import shutil
import tempfile
from . import data


class ProxyTest(LiSE.allegedb.tests.test_all.AllegedTest):
    def setUp(self):
        self.manager = EngineProcessManager()
        self.tempdir = tempfile.mkdtemp(dir='.')
        self.engine = self.manager.start(self.tempdir, connect_string='sqlite:///:memory:')
        self.graphmakers = (self.engine.new_character,)
        self.addCleanup(self._do_cleanup)

    def _do_cleanup(self):
        self.manager.shutdown()
        shutil.rmtree(self.tempdir)


class ProxyGraphTest(LiSE.allegedb.tests.test_all.AbstractGraphTest, ProxyTest):
    pass


class DictStorageTest(ProxyTest, LiSE.allegedb.tests.test_all.DictStorageTest):
    pass


class ListStorageTest(ProxyTest, LiSE.allegedb.tests.test_all.ListStorageTest):
    pass


class SetStorageTest(ProxyTest, LiSE.allegedb.tests.test_all.SetStorageTest):
    pass


@pytest.fixture(scope='function')
def handle(tempdir):
    from LiSE.handle import EngineHandle
    hand = EngineHandle((tempdir,), {'connect_string': 'sqlite:///:memory:', 'random_seed': 69105})
    yield hand
    hand.close()


@pytest.fixture(scope='function', params=[
    lambda eng: kobold.inittest(eng, shrubberies=20, kobold_sprint_chance=.9),
    # college.install,
    # sickle.install
])
def handle_initialized(request, handle):
    with handle._real.advancing():
        request.param(handle._real)
    yield handle


def test_fast_delta(handle_initialized):
    hand = handle_initialized
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


def test_assignment(handle):
    hand = handle
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


def test_serialize_deleted(engy):
    eng = engy
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


def test_manip_deleted(engy):
    eng = engy
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
