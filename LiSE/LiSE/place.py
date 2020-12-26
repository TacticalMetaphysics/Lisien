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
"""The type of node that is a location.

Though both things and places are nodes, things are obliged to be
located in another node. Places are not.

"""


from .node import Node


class Place(Node):
    """The kind of node where a thing might ultimately be located."""
    __slots__ = ('graph', 'db', 'node', '_rulebook')

    extrakeys = {
        'name',
    }

    def __getitem__(self, key):
        if key == 'name':
            return self.name
        return super().__getitem__(key)

    def __repr__(self):
        return "{}.character[{}].place[{}]".format(
            repr(self.engine),
            repr(self['character']),
            repr(self['name'])
        )

    def delete(self):
        """Remove myself from the world model immediately."""
        super().delete()
        self.character.place.send(self.character.place, key=self.name, val=None)
