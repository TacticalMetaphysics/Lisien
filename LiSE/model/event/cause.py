# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
import cause_cbs


class Cause(object):
    """Listen to the world, and fire an event on a specified condition.

In fact it is Implicator that fires the event. I only test whether a
particular precondition for that event holds true.

A Cause must have a method called test, which returns True when the
Cause is triggered. You may also provide the methods validate_time and
validate_character, which return True when the game time and the
character, respectively, are in a state such that the Cause is
*possible*, but not necessarily triggered. Use these to save on any
expensive computations that might go on in the test method.

    """

    def __init__(self, imp, test=None, validate_time=None, validate_char=None):
        """Instantiate a Cause object, specifying your own functions for the
test method and the validators.

Beware, Causes need to have a class of their own to be saveable in the
database.

        """
        self.implicator = imp
        if test is not None:
            if test.__class__ in (str, unicode):
                self.test = imp.closet.make_test(test)
            else:
                self.test = test
        if validate_time is not None:
            if validate_time.__class__ in (str, unicode):
                self.validate_time = imp.closet.make_test(validate_time)
            else:
                self.validate_time = validate_time
        if validate_char is not None:
            if validate_char is not None:
                if validate_char.__class__ in (str, unicode):
                    self.validate_char = imp.closet.make_test(validate_char)
                else:
                    self.validate_char = validate_char
        if not callable(self):
            raise TypeError(
                "Cause is not callable. Does it have a test() method?")

    def __call__(self, character, branch, tick,
                 validate_time=True, validate_char=True):
        if validate_time and not self.validate_time(branch, tick):
            return False
        if validate_char and not self.validate_character(character):
            return False
        r = self.test(character, branch, tick)
        if not isinstance(r, bool):
            raise TypeError(
                "test returned non-Boolean")
        return r

    def tester(self, cb):
        if cb.__class__ in (unicode, str):
            self.test = self.imp.closet.get_test(cb)
        else:
            self.test = cb

    def time_validator(self, cb):
        if cb.__class__ in (unicode, str):
            self.validate_time = self.imp.closet.get_test(cb)
        else:
            self.validate_time = cb

    def char_validator(self, cb):
        if cb.__class__ in (unicode, str):
            self.validate_character = self.imp.closet.get_test(cb)
        else:
            self.validate_character = cb

    def validate_time(self, branch, tick):
        return True

    def validate_character(self, character):
        return True


class Skill(Cause):
    """A Cause that a Character, possibly player-controlled, may trigger
at will.

If the skill is sometimes useless or ineffective, have its Effect(s)
do nothing under those conditions. If a Character cannot fire a
SkillCause, it does not have that skill.

    """
    def __init__(self, imp):
        super(Skill, self).__init__(imp)

    def test(self, character, branch, tick):
        if self not in character.skilldict:
            return False
        if branch not in character.skilldict[self]:
            return False
        for (tick_from, rd) in character.skilldict[self][branch]:
            tick_to = rd["tick_to"]
            if tick_from <= tick and (
                    tick_to is None or tick_to >= tick):
                return True
        return False


class Not(Cause):
    """Inverts the sense of the test--but not of the validation."""
    def __call__(self, character, branch, tick,
                 validate_time=True,
                 validate_char=True):
        return not super(Not, self).__call__(
            character, branch, tick, validate_time, validate_char)


class Multi(Cause):
    """Combine many causes into one."""
    def __init__(self, imp, causes):
        self.implicator = imp
        self.testers = []
        self.time_validators = []
        self.character_validators = []
        # All the causes are classes, names of classes, or instances.
        # Not names of instances.
        for cause in causes:
            if cause in (str, unicode):
                cause = self.implicator.make_cause(cause)
            # Possibly the methods of cause still have a self arg on
            # the beginning?
            self.testers.append(cause.test)
            self.time_validators.append(cause.validate_time)
            self.character_validators.append(cause.validate_character)


class And(Multi):
    """Validate only when all my validators pass. Trigger only when all my
tests pass."""
    def validate_time(self, branch, tick):
        for tv in self.time_validators:
            if not tv(branch, tick):
                return False
        return True

    def validate_character(self, character):
        for cv in self.character_validators:
            if not cv(character):
                return False
        return True

    def test(self, character, branch, tick):
        for t in self.testers:
            if not t(character, branch, tick):
                return False
        return True


class Or(Multi):
    """Trigger when at least one of my causes validates and passes."""
    def validate_time(self, branch, tick):
        for tv in self.time_validators:
            if tv(self, branch, tick):
                return True
        return False

    def validate_character(self, character):
        for cv in self.character_validators:
            if cv(self, character):
                return True
        return False

    def test(self, character, branch, tick):
        for t in self.testers:
            if t(character, branch, tick):
                return True
        return False


class Xor(Multi):
    """Trigger when exactly one of my causes validates and passes."""
    def validate_time(self, branch, tick):
        r = False
        for tv in self.time_validators:
            if tv(branch, tick):
                if r:
                    return False
                else:
                    r = True
        return r

    def validate_character(self, character):
        r = False
        for cv in self.character_validators:
            if cv(character):
                if r:
                    return False
                else:
                    r = True
        return r


class Nor(Or):
    """Validate the same way as Or, but only pass the test if none of the
component tests return True."""
    def test(self, character, branch, tick):
        return not super(Nor, self).test(character, branch, tick)
