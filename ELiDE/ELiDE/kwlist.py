# This file is part of ELiDE, frontend to LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  public@zacharyspector.com
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
