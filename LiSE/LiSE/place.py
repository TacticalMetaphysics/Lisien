# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""The type of node that is a location.

Though both things and places are nodes, things are obliged to be
located in another node. Places are not.

"""


from .node import Node


class Place(Node):
    """The kind of node where a thing might ultimately be located."""
    extrakeys = {
        'name',
        'character'
    }

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            return {'name': self.name, 'character': self.character.name}[key]

    def __repr__(self):
        return "{}.place[{}]".format(
            self['character'],
            self['name']
        )

    def _get_json_dict(self):
        (branch, tick) = self.engine.time
        return {
            "type": "Place",
            "version": 0,
            "character": self["character"],
            "name": self["name"],
            "branch": branch,
            "tick": tick,
            "stat": dict(self)
        }

    def dump(self):
        """Return a JSON representation of my present state"""
        return self.engine.json_dump(self._get_json_dict())

    def delete(self, nochar=False):
        """Remove myself from the world model immediately.

        With ``nochar=True``, avoid the final step of removing myself
        from my character's ``place`` mapping.

        """
        super().delete()
        if not nochar:
            del self.character.place[self.name]

    def __eq__(self, other):
        return isinstance(other, Place) and self.character.name == other.character.name and self.name == other.name
