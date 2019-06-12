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
import re
from functools import reduce
from collections import defaultdict
from LiSE.engine import Engine
import pytest
import os
import tempfile


@pytest.fixture
def college24_premade():
    codefiles = ('trigger.py', 'prereq.py', 'action.py', 'method.py', 'function.py', 'strings.json')
    tempdir = tempfile.mkdtemp(dir='.')
    for codefile in codefiles:
        if os.path.exists(codefile):
            os.rename(codefile, os.path.join(tempdir, codefile))
    if not os.path.exists("college24_premade.db"):
        print("generating test data for query")
        from LiSE.examples.college import install
        with Engine('college24_premade.db', random_seed=69105) as eng:
            install(eng)
            for i in range(24):
                print(i)
                eng.next_turn()
    eng = Engine("college24_premade.db")
    yield eng
    eng.close()
    for codefile in codefiles:
        if os.path.exists(os.path.join(tempdir, codefile)):
            os.rename(os.path.join(tempdir, codefile), codefile)
    os.rmdir(tempdir)


def roommate_collisions(engine):
    """Test queries' ability to tell that all of the students that share
    rooms have been in the same place.

    """
    done = set()
    for chara in engine.character.values():
        if chara.name in done:
            continue
        match = re.match('dorm(\d)room(\d)student(\d)', chara.name)
        if not match:
            continue
        dorm, room, student = match.groups()
        other_student = '1' if student == '0' else '0'
        student = chara
        other_student = engine.character[
            'dorm{}room{}student{}'.format(dorm, room, other_student)
        ]

        same_loc_turns = list(engine.turns_when(
            student.avatar.only.historical('location')
            == other_student.avatar.only.historical('location')
        ))
        assert same_loc_turns, "{} and {} don't seem to share a room".format(
                student.name, other_student.name
            )
        assert len(same_loc_turns) >= 6, "{} and {} did not share their room for at least 6 turns".format(
                student.name, other_student.name
            )

        done.add(student.name)
        done.add(other_student.name)


def test_roomie_collisions_premade(college24_premade):
    roommate_collisions(college24_premade)


def sober_collisions(engine):
    """Students that are neither lazy nor drunkards should all have been
    in class together at least once.

    """
    students = [
        stu for stu in
        engine.character['student_body'].stat['characters']
        if not (stu.stat['drunkard'] or stu.stat['lazy'])
    ]

    assert students

    def sameClasstime(stu0, stu1):
        assert list(
            engine.turns_when(
                stu0.avatar.only.historical('location') ==
                stu1.avatar.only.historical('location') ==
                engine.alias('classroom')
            )), """{stu0} seems not to have been in the classroom 
                at the same time as {stu1}.
                {stu0} was there at turns {turns0}
                {stu1} was there at turns {turns1}""".format(
                stu0=stu0.name,
                stu1=stu1.name,
                turns0=list(engine.turns_when(stu0.avatar.only.historical('location') == engine.alias('classroom'))),
                turns1=list(engine.turns_when(stu1.avatar.only.historical('location') == engine.alias('classroom')))
            )
        return stu1

    reduce(sameClasstime, students)


def test_sober_collisions_premade(college24_premade):
    sober_collisions(college24_premade)


def noncollision(engine):
    """Make sure students *not* from the same room never go there together"""
    dorm = defaultdict(lambda: defaultdict(dict))
    for character in engine.character.values():
        match = re.match('dorm(\d)room(\d)student(\d)', character.name)
        if not match:
            continue
        d, r, s = match.groups()
        dorm[d][r][s] = character
    for d in dorm:
        other_dorms = [dd for dd in dorm if dd != d]
        for r in dorm[d]:
            other_rooms = [rr for rr in dorm[d] if rr != r]
            for stu0 in dorm[d][r].values():
                for rr in other_rooms:
                    for stu1 in dorm[d][rr].values():
                        assert not list(engine.turns_when(
                                stu0.avatar.only.historical('location') ==
                                stu1.avatar.only.historical('location') ==
                                engine.alias('dorm{}room{}'.format(d, r))
                        )), "{} seems to share a room with {}".format(
                            stu0.name, stu1.name
                        )
                common = 'common{}'.format(d)
                for dd in other_dorms:
                    for rr in dorm[dd]:
                        for stu1 in dorm[dd][rr].values():
                            assert not list(engine.turns_when(
                                    stu0.avatar.only.historical('location') ==
                                    stu1.avatar.only.historical('location') ==
                                    engine.alias(common)
                            )), "{} seems to have been in the same common room  as {}".format(
                                stu0.name, stu1.name
                            )


def test_noncollision_premade(college24_premade):
    noncollision(college24_premade)