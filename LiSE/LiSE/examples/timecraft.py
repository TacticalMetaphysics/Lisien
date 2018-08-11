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
"""This is an interpreter for Benjamin Lortimer's Timecraft crafting ruleset.

It could be useful if you want a similar crafting system in your game; this way,
you can describe it in the same tabular format.

"""
from LiSE.rule import Rule


def install(engine):
    @engine.trigger
    def verbing(engine, character, verb):
        return character.stat.get('verb', None) == verb

    @engine.action
    def craft_thing_dict(engine, character, thing_dict):
        d = thing_dict
        body = character.avatar.physical
        num_of_kind = 0
        for node in character.node:
            if node.name == d['kind'] + str(num_of_kind):
                num_of_kind += 1
        name = d['kind'] + str(num_of_kind)
        body.new_thing(name, **d)

    @engine.trigger
    @engine.prereq
    def have_number_of_kind(engine, character, number, kind):
        body = character.avatar.physical
        return len(
            thing for thing in body.contents()
            if thing.get('kind', None) == kind
        ) >= number

    @engine.action
    def destroy_number_of_kind(engine, character, number, kind):
        body = character.avatar.physical
        n = 0
        for thing in body.contents():
            if n > number:
                break
            if thing['kind'] != kind:
                continue
            thing.delete()
            n += 1

    @engine.prereq
    def have_skill(engine, character, skill, level=None):
        if level:
            return character.skills[skill] >= level
        else:
            return skill in character.skills

    @engine.prereq
    def have_energy(engine, character, energy):
        return character.stat['energy'] >= energy

    @engine.action
    def deduct_energy(engine, character, energy):
        character.stat['energy'] -= energy

    def make_timecraft_rule(
            engine, name,
            energy=10, thing_dicts_to_add=[], material_reqs=[], skill_reqs=[]
    ):
        material_prereqs = []
        material_destroyers = []
        for req in material_reqs:
            if isinstance(req, tuple):
                if not (
                        len(req) == 2 and
                        isinstance(req[0], int) and
                        isinstance(req[1], str)
                ):
                    raise ValueError("Illegal material req: {}".format(req))
                material_prereqs.append(engine.prereq.partial(
                    'have_number_of_kind', number=req[0], kind=req[1]
                ))
                material_destroyers.append(engine.action.partial(
                    'destroy_number_of_kind', number=req[0], kind=req[1]
                ))
            elif isinstance(req, str):
                material_prereqs.append(engine.prereq.partial(
                    'have_number_of_kind', number=1, kind=req
                ))
                material_destroyers.append(engine.action.partial(
                    'destroy_number_of_kind', number=1, kind=req
                ))
            else:
                raise ValueError("Illegal material req: {}".format(req))
        skill_prereqs = []
        for req in skill_reqs:
            if isinstance(req, tuple):
                if not (
                        len(req) == 2 and
                        isinstance(req[0], int) and
                        isinstance(req[1], str)
                ):
                    raise ValueError("Illegal skill req: {}".format(req))
                skill_prereqs.append(engine.prereq.partial(
                    'have_skill', level=req[0], skill=req[1]
                ))
            elif isinstance(req, str):
                skill_prereqs.append(engine.prereq.partial(
                    'have_skill', skill=req
                ))
            else:
                raise ValueError("Illegal skill req: {}".format(req))
        return Rule(
            engine,
            name,
            triggers=[
                engine.trigger.partial(
                    'verbing', action=name
                )
            ],
            prereqs=material_prereqs + skill_prereqs,
            actions=[
                engine.action.partial(
                    'craft_thing_dict', thing_dict=thingd
                ) for thingd in thing_dicts_to_add
            ] + material_destroyers
        )


if __name__ == '__main__':
    import LiSE
    engine = LiSE.Engine('LiSEworld.db', 'LiSEcode.db')
    install(engine)
