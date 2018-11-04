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
import allegedb.tests.test_all
import pytest
import LiSE.examples.kobold as kobold
import LiSE.examples.college as college
import LiSE.examples.sickle as sickle
import os
import tempfile


class ProxyTest(allegedb.tests.test_all.AllegedTest):
    def setUp(self):
        self.manager = EngineProcessManager()
        self.engine = self.manager.start('sqlite:///:memory:')
        self.graphmakers = (self.engine.new_character,)
        self.tempdir = tempfile.mkdtemp(dir='.')
        for f in (
                'trigger.py', 'prereq.py', 'action.py', 'function.py',
                'method.py', 'strings.json'
        ):
            if os.path.exists(f):
                os.rename(f, os.path.join(self.tempdir, f))

    def tearDown(self):
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


class ProxyGraphTest(allegedb.tests.test_all.AbstractGraphTest, ProxyTest):
    pass


class DictStorageTest(ProxyTest, allegedb.tests.test_all.DictStorageTest):
    pass


class ListStorageTest(ProxyTest, allegedb.tests.test_all.ListStorageTest):
    pass


class SetStorageTest(ProxyTest, allegedb.tests.test_all.SetStorageTest):
    pass


@pytest.fixture(scope='function', params=[
    lambda eng: kobold.inittest(eng, shrubberies=20, kobold_sprint_chance=.9),
    college.install,
    sickle.install
])
def hand(request):
    from LiSE.handle import EngineHandle
    tempdir = tempfile.mkdtemp(dir='.')
    for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
    ):
        if os.path.exists(f):
            os.rename(f, os.path.join(tempdir, f))
    hand = EngineHandle((':memory:',), {'random_seed': 69105})
    with hand._real.advancing():
        request.param(hand._real)
    yield hand
    hand.close()
    for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
    ):
        if os.path.exists(os.path.join(tempdir, f)):
            os.rename(os.path.join(tempdir, f), f)
    os.rmdir(tempdir)


def test_fast_delta(hand):
    # just set a baseline for the diff
    hand.get_slow_delta()
    ret, diff = hand.next_turn()
    slowd = hand.get_slow_delta()
    assert diff == slowd, "Fast delta differs from slow delta"
    ret, diff2 = hand.time_travel('trunk', 0, 0)
    slowd2 = hand.get_slow_delta()
    assert diff2 == slowd2, "Fast delta differs from slow delta"
    ret, diff3 = hand.time_travel('trunk', 3)
    slowd3 = hand.get_slow_delta()
    assert diff3 == slowd3, "Fast delta differs from slow delta"
    ret, diff4 = hand.time_travel('trunk', 1)
    slowd4 = hand.get_slow_delta()
    assert diff4 == slowd4, "Fast delta differs from slow delta"


def test_assignment():
    from LiSE.handle import EngineHandle
    tempdir = tempfile.mkdtemp(dir='.')
    for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
    ):
        if os.path.exists(f):
            os.rename(f, os.path.join(tempdir, f))
    hand = EngineHandle((':memory:',), {'random_seed': 69105})
    eng = hand._real
    with eng.advancing():
        college.install(eng)
    physical_inital_copy = {'node_val': {'common0': {'rulebook': ('physical', 'common0')},
                                         'dorm1room3': {'rulebook': ('physical', 'dorm1room3')},
                                         'dorm1room4student1': {'rulebook': ('physical', 'dorm1room4student1'),
                                                                'location': 'dorm1room4'},
                                         'dorm2room4student0': {'rulebook': ('physical', 'dorm2room4student0'),
                                                                'location': 'dorm2room4'},
                                         'classroom': {'rulebook': ('physical', 'classroom')},
                                         'dorm0room5student1': {'rulebook': ('physical', 'dorm0room5student1'),
                                                                'location': 'dorm0room5'},
                                         'dorm2room3student1': {'rulebook': ('physical', 'dorm2room3student1'),
                                                                'location': 'dorm2room3'},
                                         'dorm0room3student0': {'rulebook': ('physical', 'dorm0room3student0'),
                                                                'location': 'dorm0room3'},
                                         'dorm0room2student0': {'rulebook': ('physical', 'dorm0room2student0'),
                                                                'location': 'dorm0room2'},
                                         'dorm2room1student0': {'rulebook': ('physical', 'dorm2room1student0'),
                                                                'location': 'dorm2room1'},
                                         'dorm1room4student0': {'rulebook': ('physical', 'dorm1room4student0'),
                                                                'location': 'dorm1room4'},
                                         'dorm2room0student1': {'rulebook': ('physical', 'dorm2room0student1'),
                                                                'location': 'dorm2room0'},
                                         'dorm1room1': {'rulebook': ('physical', 'dorm1room1')},
                                         'dorm2room0': {'rulebook': ('physical', 'dorm2room0')},
                                         'dorm2room3student0': {'rulebook': ('physical', 'dorm2room3student0'),
                                                                'location': 'dorm2room3'},
                                         'dorm0room0student0': {'rulebook': ('physical', 'dorm0room0student0'),
                                                                'location': 'dorm0room0'},
                                         'dorm2room2student0': {'rulebook': ('physical', 'dorm2room2student0'),
                                                                'location': 'dorm2room2'},
                                         'dorm0room3student1': {'rulebook': ('physical', 'dorm0room3student1'),
                                                                'location': 'dorm0room3'},
                                         'dorm0room5': {'rulebook': ('physical', 'dorm0room5')},
                                         'dorm2room1': {'rulebook': ('physical', 'dorm2room1')},
                                         'dorm1room1student1': {'rulebook': ('physical', 'dorm1room1student1'),
                                                                'location': 'dorm1room1'},
                                         'dorm0room4student0': {'rulebook': ('physical', 'dorm0room4student0'),
                                                                'location': 'dorm0room4'},
                                         'dorm2room2student1': {'rulebook': ('physical', 'dorm2room2student1'),
                                                                'location': 'dorm2room2'},
                                         'dorm1room2': {'rulebook': ('physical', 'dorm1room2')},
                                         'dorm0room2': {'rulebook': ('physical', 'dorm0room2')},
                                         'dorm2room5': {'rulebook': ('physical', 'dorm2room5')},
                                         'dorm1room3student1': {'rulebook': ('physical', 'dorm1room3student1'),
                                                                'location': 'dorm1room3'},
                                         'dorm0room2student1': {'rulebook': ('physical', 'dorm0room2student1'),
                                                                'location': 'dorm0room2'},
                                         'dorm1room5student0': {'rulebook': ('physical', 'dorm1room5student0'),
                                                                'location': 'dorm1room5'},
                                         'dorm1room0student0': {'rulebook': ('physical', 'dorm1room0student0'),
                                                                'location': 'dorm1room0'},
                                         'dorm2room5student0': {'rulebook': ('physical', 'dorm2room5student0'),
                                                                'location': 'dorm2room5'},
                                         'dorm1room5student1': {'rulebook': ('physical', 'dorm1room5student1'),
                                                                'location': 'dorm1room5'},
                                         'dorm1room4': {'rulebook': ('physical', 'dorm1room4')},
                                         'dorm0room1student1': {'rulebook': ('physical', 'dorm0room1student1'),
                                                                'location': 'dorm0room1'},
                                         'dorm1room2student0': {'rulebook': ('physical', 'dorm1room2student0'),
                                                                'location': 'dorm1room2'},
                                         'common2': {'rulebook': ('physical', 'common2')},
                                         'dorm2room1student1': {'rulebook': ('physical', 'dorm2room1student1'),
                                                                'location': 'dorm2room1'},
                                         'dorm1room0student1': {'rulebook': ('physical', 'dorm1room0student1'),
                                                                'location': 'dorm1room0'},
                                         'dorm1room2student1': {'rulebook': ('physical', 'dorm1room2student1'),
                                                                'location': 'dorm1room2'},
                                         'dorm0room0student1': {'rulebook': ('physical', 'dorm0room0student1'),
                                                                'location': 'dorm0room0'},
                                         'dorm0room1student0': {'rulebook': ('physical', 'dorm0room1student0'),
                                                                'location': 'dorm0room1'},
                                         'dorm1room0': {'rulebook': ('physical', 'dorm1room0')},
                                         'dorm2room0student0': {'rulebook': ('physical', 'dorm2room0student0'),
                                                                'location': 'dorm2room0'},
                                         'dorm0room4': {'rulebook': ('physical', 'dorm0room4')},
                                         'dorm0room4student1': {'rulebook': ('physical', 'dorm0room4student1'),
                                                                'location': 'dorm0room4'},
                                         'dorm2room3': {'rulebook': ('physical', 'dorm2room3')},
                                         'dorm0room1': {'rulebook': ('physical', 'dorm0room1')},
                                         'dorm1room1student0': {'rulebook': ('physical', 'dorm1room1student0'),
                                                                'location': 'dorm1room1'},
                                         'dorm0room0': {'rulebook': ('physical', 'dorm0room0')},
                                         'dorm1room3student0': {'rulebook': ('physical', 'dorm1room3student0'),
                                                                'location': 'dorm1room3'},
                                         'dorm2room2': {'rulebook': ('physical', 'dorm2room2')},
                                         'dorm2room4student1': {'rulebook': ('physical', 'dorm2room4student1'),
                                                                'location': 'dorm2room4'},
                                         'dorm0room5student0': {'rulebook': ('physical', 'dorm0room5student0'),
                                                                'location': 'dorm0room5'},
                                         'dorm1room5': {'rulebook': ('physical', 'dorm1room5')},
                                         'dorm0room3': {'rulebook': ('physical', 'dorm0room3')},
                                         'dorm2room5student1': {'rulebook': ('physical', 'dorm2room5student1'),
                                                                'location': 'dorm2room5'},
                                         'common1': {'rulebook': ('physical', 'common1')},
                                         'dorm2room4': {'rulebook': ('physical', 'dorm2room4')}},
                            'hour': 0,
                            'name': 'physical',
                            'edge_val': {'common0': {'dorm0room5': {'is_mirror': True},
                                                     'dorm0room3': {'is_mirror': True},
                                                     'classroom': {},
                                                     'dorm0room1': {'is_mirror': True},
                                                     'dorm0room0': {'is_mirror': True},
                                                     'dorm0room4': {'is_mirror': True},
                                                     'dorm0room2': {'is_mirror': True}},
                                         'dorm1room3': {'common1': {}},
                                         'common2': {'dorm2room1': {'is_mirror': True},
                                                     'dorm2room2': {'is_mirror': True},
                                                     'dorm2room0': {'is_mirror': True},
                                                     'classroom': {},
                                                     'dorm2room5': {'is_mirror': True},
                                                     'dorm2room4': {'is_mirror': True},
                                                     'dorm2room3': {'is_mirror': True}},
                                         'classroom': {'common0': {'is_mirror': True},
                                                       'common2': {'is_mirror': True},
                                                       'common1': {'is_mirror': True}},
                                         'dorm1room1': {'common1': {}},
                                         'dorm2room0': {'common2': {}},
                                         'dorm1room0': {'common1': {}},
                                         'dorm0room0': {'common0': {}},
                                         'dorm0room4': {'common0': {}},
                                         'dorm2room3': {'common2': {}},
                                         'dorm0room1': {'common0': {}},
                                         'dorm2room2': {'common2': {}},
                                         'dorm1room5': {'common1': {}},
                                         'dorm1room2': {'common1': {}},
                                         'dorm0room2': {'common0': {}},
                                         'dorm2room1': {'common2': {}},
                                         'dorm2room5': {'common2': {}},
                                         'dorm2room4': {'common2': {}},
                                         'dorm0room3': {'common0': {}},
                                         'dorm0room5': {'common0': {}},
                                         'common1': {'dorm1room3': {'is_mirror': True},
                                                     'dorm1room1': {'is_mirror': True},
                                                     'classroom': {},
                                                     'dorm1room5': {'is_mirror': True},
                                                     'dorm1room0': {'is_mirror': True},
                                                     'dorm1room2': {'is_mirror': True},
                                                     'dorm1room4': {'is_mirror': True}},
                                         'dorm1room4': {'common1': {}}},
                            'rulebooks': {'character': ('physical', 'character'),
                                          'thing': ('physical', 'character_thing'),
                                          'avatar': ('physical', 'avatar'),
                                          'place': ('physical', 'character_place'),
                                          'portal': ('physical', 'character_portal')}}
    physical_copy = hand.character_copy('physical')
    assert physical_copy == physical_inital_copy
    dorm_initial_copy = {'rulebooks': {'thing': ('dorm0', 'character_thing'), 'character': ('dorm0', 'character'),
                                       'avatar': ('dorm0', 'avatar'),
                                       'portal': ('dorm0', 'character_portal'), 'place': ('dorm0', 'character_place')},
                         'name': 'dorm0', 'avatars': {'physical': frozenset(
            {'dorm0room1', 'dorm0room2', 'dorm0room4', 'dorm0room5', 'dorm0room3', 'common0', 'dorm0room0'})}}
    dorm_copy = hand.character_copy('dorm0')
    assert dorm_copy == dorm_initial_copy
    student_initial_copy = {'roommate': eng.character['dorm0room0student1'],
                            'avatars': {'physical': frozenset({'dorm0room0student0'})}, 'xp': 0, 'lazy': True,
                            'drunkard': False, 'room': eng.character['physical'].place['dorm0room0'],
                            'name': 'dorm0room0student0', 'rulebooks': {
            'place': ('dorm0room0student0', 'character_place'), 'avatar': ('dorm0room0student0', 'avatar'),
            'portal': ('dorm0room0student0', 'character_portal'), 'thing': ('dorm0room0student0', 'character_thing'),
            'character': ('dorm0room0student0', 'character')}, 'node_val': {
            'cell26': {'rulebook': ('dorm0room0student0', 'cell26'), 'drunk': 0, 'slow': 0},
            'cell13': {'rulebook': ('dorm0room0student0', 'cell13'), 'drunk': 0, 'slow': 0},
            'cell86': {'rulebook': ('dorm0room0student0', 'cell86'), 'drunk': 0, 'slow': 0},
            'cell11': {'rulebook': ('dorm0room0student0', 'cell11'), 'drunk': 0, 'slow': 0},
            'cell24': {'rulebook': ('dorm0room0student0', 'cell24'), 'drunk': 0, 'slow': 0},
            'cell53': {'rulebook': ('dorm0room0student0', 'cell53'), 'drunk': 0, 'slow': 0},
            'cell60': {'rulebook': ('dorm0room0student0', 'cell60'), 'drunk': 0, 'slow': 0},
            'cell55': {'rulebook': ('dorm0room0student0', 'cell55'), 'drunk': 0, 'slow': 0},
            'cell51': {'rulebook': ('dorm0room0student0', 'cell51'), 'drunk': 0, 'slow': 0},
            'cell25': {'rulebook': ('dorm0room0student0', 'cell25'), 'drunk': 0, 'slow': 0},
            'cell98': {'rulebook': ('dorm0room0student0', 'cell98'), 'drunk': 0, 'slow': 0},
            'cell85': {'rulebook': ('dorm0room0student0', 'cell85'), 'drunk': 0, 'slow': 0},
            'cell81': {'rulebook': ('dorm0room0student0', 'cell81'), 'drunk': 0, 'slow': 0},
            'cell63': {'rulebook': ('dorm0room0student0', 'cell63'), 'drunk': 0, 'slow': 0},
            'cell47': {'rulebook': ('dorm0room0student0', 'cell47'), 'drunk': 0, 'slow': 0},
            'cell78': {'rulebook': ('dorm0room0student0', 'cell78'), 'drunk': 0, 'slow': 0},
            'cell0': {'rulebook': ('dorm0room0student0', 'cell0'), 'drunk': 0, 'slow': 0},
            'cell12': {'rulebook': ('dorm0room0student0', 'cell12'), 'drunk': 0, 'slow': 0},
            'cell35': {'rulebook': ('dorm0room0student0', 'cell35'), 'drunk': 0, 'slow': 0},
            'cell99': {'rulebook': ('dorm0room0student0', 'cell99'), 'drunk': 0, 'slow': 0},
            'cell16': {'rulebook': ('dorm0room0student0', 'cell16'), 'drunk': 0, 'slow': 0},
            'cell57': {'rulebook': ('dorm0room0student0', 'cell57'), 'drunk': 0, 'slow': 0},
            'cell92': {'rulebook': ('dorm0room0student0', 'cell92'), 'drunk': 0, 'slow': 0},
            'cell1': {'rulebook': ('dorm0room0student0', 'cell1'), 'drunk': 0, 'slow': 0},
            'cell62': {'rulebook': ('dorm0room0student0', 'cell62'), 'drunk': 0, 'slow': 0},
            'cell80': {'rulebook': ('dorm0room0student0', 'cell80'), 'drunk': 0, 'slow': 0},
            'cell44': {'rulebook': ('dorm0room0student0', 'cell44'), 'drunk': 0, 'slow': 0},
            'cell82': {'rulebook': ('dorm0room0student0', 'cell82'), 'drunk': 0, 'slow': 0},
            'cell37': {'rulebook': ('dorm0room0student0', 'cell37'), 'drunk': 0, 'slow': 0},
            'cell20': {'rulebook': ('dorm0room0student0', 'cell20'), 'drunk': 0, 'slow': 0},
            'cell87': {'rulebook': ('dorm0room0student0', 'cell87'), 'drunk': 0, 'slow': 0},
            'cell64': {'rulebook': ('dorm0room0student0', 'cell64'), 'drunk': 0, 'slow': 0},
            'cell3': {'rulebook': ('dorm0room0student0', 'cell3'), 'drunk': 0, 'slow': 0},
            'cell23': {'rulebook': ('dorm0room0student0', 'cell23'), 'drunk': 0, 'slow': 0},
            'cell28': {'rulebook': ('dorm0room0student0', 'cell28'), 'drunk': 0, 'slow': 0},
            'cell95': {'rulebook': ('dorm0room0student0', 'cell95'), 'drunk': 0, 'slow': 0},
            'cell59': {'rulebook': ('dorm0room0student0', 'cell59'), 'drunk': 0, 'slow': 0},
            'cell91': {'rulebook': ('dorm0room0student0', 'cell91'), 'drunk': 0, 'slow': 0},
            'cell34': {'rulebook': ('dorm0room0student0', 'cell34'), 'drunk': 0, 'slow': 0},
            'cell32': {'rulebook': ('dorm0room0student0', 'cell32'), 'drunk': 0, 'slow': 0},
            'cell31': {'rulebook': ('dorm0room0student0', 'cell31'), 'drunk': 0, 'slow': 0},
            'cell21': {'rulebook': ('dorm0room0student0', 'cell21'), 'drunk': 0, 'slow': 0},
            'cell4': {'rulebook': ('dorm0room0student0', 'cell4'), 'drunk': 0, 'slow': 0},
            'cell77': {'rulebook': ('dorm0room0student0', 'cell77'), 'drunk': 0, 'slow': 0},
            'cell27': {'rulebook': ('dorm0room0student0', 'cell27'), 'drunk': 0, 'slow': 0},
            'cell61': {'rulebook': ('dorm0room0student0', 'cell61'), 'drunk': 0, 'slow': 0},
            'cell70': {'rulebook': ('dorm0room0student0', 'cell70'), 'drunk': 0, 'slow': 0},
            'cell84': {'rulebook': ('dorm0room0student0', 'cell84'), 'drunk': 0, 'slow': 0},
            'cell36': {'rulebook': ('dorm0room0student0', 'cell36'), 'drunk': 0, 'slow': 0},
            'cell30': {'rulebook': ('dorm0room0student0', 'cell30'), 'drunk': 0, 'slow': 0},
            'cell88': {'rulebook': ('dorm0room0student0', 'cell88'), 'drunk': 0, 'slow': 0},
            'cell52': {'rulebook': ('dorm0room0student0', 'cell52'), 'drunk': 0, 'slow': 0},
            'cell58': {'rulebook': ('dorm0room0student0', 'cell58'), 'drunk': 0, 'slow': 0},
            'cell2': {'rulebook': ('dorm0room0student0', 'cell2'), 'drunk': 0, 'slow': 0},
            'cell75': {'rulebook': ('dorm0room0student0', 'cell75'), 'drunk': 0, 'slow': 0},
            'cell69': {'rulebook': ('dorm0room0student0', 'cell69'), 'drunk': 0, 'slow': 0},
            'cell74': {'rulebook': ('dorm0room0student0', 'cell74'), 'drunk': 0, 'slow': 0},
            'cell49': {'rulebook': ('dorm0room0student0', 'cell49'), 'drunk': 0, 'slow': 0},
            'cell90': {'rulebook': ('dorm0room0student0', 'cell90'), 'drunk': 0, 'slow': 0},
            'cell15': {'rulebook': ('dorm0room0student0', 'cell15'), 'drunk': 0, 'slow': 0},
            'cell65': {'rulebook': ('dorm0room0student0', 'cell65'), 'drunk': 0, 'slow': 0},
            'cell50': {'rulebook': ('dorm0room0student0', 'cell50'), 'drunk': 0, 'slow': 0},
            'cell22': {'rulebook': ('dorm0room0student0', 'cell22'), 'drunk': 0, 'slow': 0},
            'cell46': {'rulebook': ('dorm0room0student0', 'cell46'), 'drunk': 0, 'slow': 0},
            'cell93': {'rulebook': ('dorm0room0student0', 'cell93'), 'drunk': 0, 'slow': 0},
            'cell40': {'rulebook': ('dorm0room0student0', 'cell40'), 'drunk': 0, 'slow': 0},
            'cell66': {'rulebook': ('dorm0room0student0', 'cell66'), 'drunk': 0, 'slow': 0},
            'cell73': {'rulebook': ('dorm0room0student0', 'cell73'), 'drunk': 0, 'slow': 0},
            'cell8': {'rulebook': ('dorm0room0student0', 'cell8'), 'drunk': 0, 'slow': 0},
            'cell41': {'rulebook': ('dorm0room0student0', 'cell41'), 'drunk': 0, 'slow': 0},
            'cell17': {'rulebook': ('dorm0room0student0', 'cell17'), 'drunk': 0, 'slow': 0},
            'cell56': {'rulebook': ('dorm0room0student0', 'cell56'), 'drunk': 0, 'slow': 0},
            'cell48': {'rulebook': ('dorm0room0student0', 'cell48'), 'drunk': 0, 'slow': 0},
            'cell89': {'rulebook': ('dorm0room0student0', 'cell89'), 'drunk': 0, 'slow': 0},
            'cell5': {'rulebook': ('dorm0room0student0', 'cell5'), 'drunk': 0, 'slow': 0},
            'cell94': {'rulebook': ('dorm0room0student0', 'cell94'), 'drunk': 0, 'slow': 0},
            'cell19': {'rulebook': ('dorm0room0student0', 'cell19'), 'drunk': 0, 'slow': 0},
            'cell71': {'rulebook': ('dorm0room0student0', 'cell71'), 'drunk': 0, 'slow': 0},
            'cell67': {'rulebook': ('dorm0room0student0', 'cell67'), 'drunk': 0, 'slow': 0},
            'cell96': {'rulebook': ('dorm0room0student0', 'cell96'), 'drunk': 0, 'slow': 0},
            'cell18': {'rulebook': ('dorm0room0student0', 'cell18'), 'drunk': 0, 'slow': 0},
            'cell42': {'rulebook': ('dorm0room0student0', 'cell42'), 'drunk': 0, 'slow': 0},
            'cell38': {'rulebook': ('dorm0room0student0', 'cell38'), 'drunk': 0, 'slow': 0},
            'cell29': {'rulebook': ('dorm0room0student0', 'cell29'), 'drunk': 0, 'slow': 0},
            'cell54': {'rulebook': ('dorm0room0student0', 'cell54'), 'drunk': 0, 'slow': 0},
            'cell14': {'rulebook': ('dorm0room0student0', 'cell14'), 'drunk': 0, 'slow': 0},
            'cell9': {'rulebook': ('dorm0room0student0', 'cell9'), 'drunk': 0, 'slow': 0},
            'cell33': {'rulebook': ('dorm0room0student0', 'cell33'), 'drunk': 0, 'slow': 0},
            'cell68': {'rulebook': ('dorm0room0student0', 'cell68'), 'drunk': 0, 'slow': 0},
            'cell10': {'rulebook': ('dorm0room0student0', 'cell10'), 'drunk': 0, 'slow': 0},
            'cell7': {'rulebook': ('dorm0room0student0', 'cell7'), 'drunk': 0, 'slow': 0},
            'cell76': {'rulebook': ('dorm0room0student0', 'cell76'), 'drunk': 0, 'slow': 0},
            'cell45': {'rulebook': ('dorm0room0student0', 'cell45'), 'drunk': 0, 'slow': 0},
            'cell83': {'rulebook': ('dorm0room0student0', 'cell83'), 'drunk': 0, 'slow': 0},
            'cell43': {'rulebook': ('dorm0room0student0', 'cell43'), 'drunk': 0, 'slow': 0},
            'cell79': {'rulebook': ('dorm0room0student0', 'cell79'), 'drunk': 0, 'slow': 0},
            'cell97': {'rulebook': ('dorm0room0student0', 'cell97'), 'drunk': 0, 'slow': 0},
            'cell39': {'rulebook': ('dorm0room0student0', 'cell39'), 'drunk': 0, 'slow': 0},
            'cell6': {'rulebook': ('dorm0room0student0', 'cell6'), 'drunk': 0, 'slow': 0},
            'cell72': {'rulebook': ('dorm0room0student0', 'cell72'), 'drunk': 0, 'slow': 0}}}
    assert hand.character_copy('dorm0room0student0') == student_initial_copy
    hand.close()
    for f in (
            'trigger.py', 'prereq.py', 'action.py', 'function.py',
            'method.py', 'strings.json'
    ):
        if os.path.exists(os.path.join(tempdir, f)):
            os.rename(os.path.join(tempdir, f), f)
    os.rmdir(tempdir)
