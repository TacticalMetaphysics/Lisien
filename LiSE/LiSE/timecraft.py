# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from LiSE.rule import Rule
from operator import itemgetter


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

    @engine.function
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


def iget(i):
    return property(itemgetter(i))


class recipe(tuple):
    def __new__(cls, *args):
        if len(args) == 1:
            args = args[0]
        (
            key,
            action,
            item_id,
            name,
            kind,
            tech,
            skill,
            consumes,
            time,
            needs,
            byproducts,
            effect,
            geological,
            weight,
            turns_into
        ) = args
        return tuple.__new__(cls, (
            int(key),
            action,
            int(item_id),
            name,
            kind,
            int(tech),
            skill,
            frozenset(consumes),
            int(time) if time else None,
            frozenset(needs.split(', ')),
            frozenset(byproducts.split(', ')),
            effect,
            frozenset(geological.split(', ')),
            float(weight) if weight else None,
            turns_into
        ))

    key = iget(0)
    action = iget(1)
    item_id = iget(2)
    name = iget(3)
    kind = iget(4)
    tech = iget(5)
    skill = iget(6)
    consumes = iget(7)
    time = iget(8)
    needs = iget(9)
    byproducts = iget(10)
    effect = iget(11)
    geological = iget(12)
    weight = iget(13)
    turns_into = iget(14)


header = (
    'Id',
    'Action',
    'Item Id',
    'Name',
    'Type',
    'Tech',
    'Skill',
    'Consumes',
    'Time',
    'Needs',
    'Byproducts',
    'Effect',
    'Geological',
    'Weight',
    'Turns into'
)


def parse_timecraft(f: "file-like object") -> dict:
    from csv import reader
    from collections import defaultdict
    out = defaultdict(set)
    for line in reader(f):
        if line == header:
            continue
        out[line[0]] = recipe(line)
    return out

if __name__ == '__main__':
    import LiSE
    engine = LiSE.Engine('LiSEworld.db', 'LiSEcode.db')
    install(engine)
