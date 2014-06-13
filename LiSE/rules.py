# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
from LiSE.orm import SaveableMetaclass


class Rule(object):
    __metaclass__ = SaveableMetaclass
    tables = [
        (
            'rules',
            {
                'columns':
                [
                    {
                        'name': 'name',
                        'type': 'text'
                    }
                ],
                'primary_key': ('name',)
            }
        ), (
            "rule_prereqs",
            {
                "columns":
                [
                    {
                        "name": "rule",
                        'type': 'text'
                    }, {
                        'name': 'idx',
                        'type': 'integer'
                    }, {
                        'name': 'function',
                        'type': 'text',
                        'nullable': True
                    }
                ],
                'primary_key': ('rule', 'idx'),
                'foreign_keys': {'rule': ('rules', 'name')},
                'checks': ['idx>=0']
            }
        )
    ]

    def __init__(self, character, effect):
        if isinstance(effect, str) or isinstance(effect, unicode):
            name = effect
            effect = character.closet.shelf[name]
            assert(name == effect.__name__)
        elif effect.__name__ not in character.closet.shelf:
            name = effect.__name__
            character.closet.shelf[name] = effect
        # TODO check for effect collisions
        character.closet.set_bone(
            self.bonetypes['rules'](name=name)
        )
        self.character = character
        self.effect = effect

    def __call__(self):
        return self.effect(self.character)

    def prereq(self, prereq, idx=None):
        if isinstance(prereq, str) or isinstance(prereq, unicode):
            name = prereq
            prereq = self.character.closet.shelf[name]
        elif prereq.__name__ not in self.character.closet.shelf:
            name = prereq.__name__
            self.character.closet.shelf[name] = prereq
        if idx is None:
            try:
                idx = max(self.character.closet.skeleton['rule_prereqs'][name].viewkeys()) + 1
            except ValueError:
                idx = 0
        elif idx < 0:
            raise IndexError("No negative indices")
        # TODO check for prereq collisions
        self.character.closet.set_bone(
            self.bonetypes['rule_prereqs'](
                name=name,
                idx=idx
            )
        )

    def iter_prereqs(self):
        skel = self.character.closet.skeleton['rule_prereqs'][self.effect.__name__]
        for i in skel:
            if skel[i].value is not None:
                yield self.character.closet.shelf[skel[i].value]
