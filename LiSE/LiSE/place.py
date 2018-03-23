# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
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
        return "{}.character['{}'].place['{}']".format(
            self.engine,
            self['character'],
            self['name']
        )

    def delete(self):
        """Remove myself from the world model immediately."""
        super().delete()
        self.character.place.send(self.character.place, key=self.name, val=None)
