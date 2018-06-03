import pytest
import os
from LiSE.engine import Engine


@pytest.fixture(scope='function')
def engy():
    codefiles = ('trigger.py', 'prereq.py', 'action.py', 'method.py', 'function.py')
    for file in codefiles:
        if os.path.exists(file):
            os.remove(file)
    with Engine(":memory:") as eng:
        yield eng
    for file in codefiles:
        os.remove(file)


def something_dot_rule_test(something, eng):
    @something.rule
    def somerule():
        pass

    @somerule.trigger
    def otherthing():
        pass

    @somerule.prereq
    def anotherthing():
        pass

    assert 'somerule' in eng.rule
    assert somerule.triggers
    assert eng.trigger.otherthing in somerule.triggers
    assert somerule.prereqs
    assert eng.prereq.anotherthing in somerule.prereqs

    @somerule.trigger
    def thirdthing():
        pass

    assert somerule.triggers.index(eng.trigger.otherthing) == 0
    assert somerule.triggers.index(eng.trigger.thirdthing) == 1

    @somerule.prereq
    def fourththing():
        pass

    assert somerule.prereqs.index(eng.prereq.anotherthing) == 0
    assert somerule.prereqs.index(eng.prereq.fourththing) == 1

    @somerule.action
    def fifththing():
        pass

    assert somerule.actions.index(eng.action.somerule) == 0
    assert somerule.actions.index(eng.action.fifththing) == 1

    del somerule.triggers[0]
    del somerule.prereqs[0]
    del somerule.actions[0]
    assert somerule.triggers[0] == eng.trigger.thirdthing
    assert somerule.prereqs[0] == eng.prereq.fourththing
    assert somerule.actions[0] == eng.action.fifththing

    somerule.triggers.append('otherthing')
    somerule.prereqs.append('anotherthing')
    somerule.actions.append('somerule')
    assert somerule.triggers[1] == eng.trigger.otherthing
    assert somerule.prereqs[1] == eng.prereq.anotherthing
    assert somerule.actions[1] == eng.action.somerule


def test_engine_dot_rule(engy):
    something_dot_rule_test(engy, engy)


def test_character_dot_rule(engy):
    character = engy.new_character('physical')
    something_dot_rule_test(character, engy)
    assert character.rulebook[0] == engy.rule['somerule']


def test_character_dot_thing_dot_rule(engy):
    character = engy.new_character('physical')
    something_dot_rule_test(character.thing, engy)
    assert character.thing.rulebook[0] == engy.rule['somerule']


def test_character_dot_place_dot_rule(engy):
    character = engy.new_character('physical')
    something_dot_rule_test(character.place, engy)
    assert character.place.rulebook[0] == engy.rule['somerule']


def test_character_dot_portal_dot_rule(engy):
    character = engy.new_character('physical')
    something_dot_rule_test(character.portal, engy)
    assert character.portal.rulebook[0] == engy.rule['somerule']


def test_node_dot_rule(engy):
    here = engy.new_character('physical').new_place(1)
    something_dot_rule_test(here, engy)
    assert here.rulebook[0] == engy.rule['somerule']


def test_portal_dot_rule(engy):
    character = engy.new_character('physical')
    character.new_place(0)
    character.new_place(1)
    port = character.new_portal(0, 1)
    something_dot_rule_test(port, engy)
    assert port.rulebook[0] == engy.rule['somerule']