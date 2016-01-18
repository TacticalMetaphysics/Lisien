# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
from LiSE.rule import Rule


def install(engine):
    @engine.function
    def taking_action(engine, character, action):
        return character.stat.get('action', None) == action

    @engine.function
    def craft_thing_dict(engine, character, thing_dict):
        d = thing_dict
        body = character.avatar.physical
        num_of_kind = 0
        for node in character.node:
            if node.name == d['kind'] + str(num_of_kind):
                num_of_kind += 1
        name = d['kind'] + str(num_of_kind)
        body.new_thing(name, **d)

    @engine.function
    def have_number_of_kind(engine, character, number, kind):
        body = character.avatar.physical
        return len(
            thing for thing in body.contents()
            if thing.get('kind', None) == kind
        ) >= number

    @engine.function
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

    @engine.function
    def have_skill(engine, character, skill, level=None):
        if level:
            return character.skills[skill] >= level
        else:
            return skill in character.skills

    @engine.function
    def have_energy(engine, character, energy):
        return character.stat['energy'] >= energy

    @engine.function
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
                material_prereqs.append(engine.function.partial(
                    'have_number_of_kind', number=req[0], kind=req[1]
                ))
                material_destroyers.append(engine.function.partial(
                    'destroy_number_of_kind', number=req[0], kind=req[1]
                ))
            elif isinstance(req, str):
                material_prereqs.append(engine.function.partial(
                    'have_number_of_kind', number=1, kind=req
                ))
                material_destroyers.append(engine.function.partial(
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
                skill_prereqs.append(engine.function.partial(
                    'have_skill', level=req[0], skill=req[1]
                ))
            elif isinstance(req, str):
                skill_prereqs.append(engine.function.partial(
                    'have_skill', skill=req
                ))
            else:
                raise ValueError("Illegal skill req: {}".format(req))
        return Rule(
            engine,
            name,
            triggers=[
                engine.function.partial(
                    'taking_action', action=name
                )
            ],
            prereqs=material_prereqs + skill_prereqs,
            actions=[
                engine.function_partial(
                    'craft_thing_dict', thing_dict=thingd
                ) for thingd in thing_dicts_to_add
            ] + material_destroyers
        )
