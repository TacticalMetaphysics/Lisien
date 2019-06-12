# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
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
from kivy.lang import Builder
from kivy.uix.modalview import ModalView
from kivy.properties import ListProperty


class KeywordListModal(ModalView):
    data = ListProperty([])


Builder.load_string("""
<KeywordListModal>:
    size_hint_x: 0.6
    BoxLayout:
        orientation: 'vertical'
        StatListView:
            data: root.data
        BoxLayout:
            Button:
                text: 'Cancel'
            Button:
                text: 'Done'
""")
