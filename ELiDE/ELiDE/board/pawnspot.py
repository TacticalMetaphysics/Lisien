# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
"""Code that draws the box around a Pawn or Spot when it's selected"""
from functools import partial
from collections import defaultdict

from kivy.properties import (
    ObjectProperty,
    BooleanProperty,
    ListProperty,
    DictProperty
)
from kivy.graphics import (
    InstructionGroup,
    Color,
    Line
)
from kivy.uix.layout import Layout
from kivy.clock import Clock
from kivy.logger import Logger
from kivy.lang import Builder
from ELiDE.kivygarden.texturestack import ImageStack
from ..util import trigger


class PawnSpot(ImageStack, Layout):
    """The kind of ImageStack that represents a :class:`Thing` or
    :class:`Place`.

    """
    board = ObjectProperty()
    proxy = ObjectProperty()
    engine = ObjectProperty()
    selected = BooleanProperty(False)
    linecolor = ListProperty()
    name = ObjectProperty()
    use_boardspace = True
    positions = DictProperty()
    _childs = DictProperty()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(pos=self._position, positions=self._position)

    def on_touch_move(self, touch):
        """If I'm being dragged, move to follow the touch."""
        if touch.grab_current is not self:
            return False
        Logger.debug("PawnSpot: {} getting dragged to {}".format(self.name, touch.pos))
        self.center = touch.pos
        return True

    def finalize(self, initial=True):
        """Call this after you've created all the PawnSpot you need and are ready to add them to the board."""
        if getattr(self, '_finalized', False):
            return
        if (
                self.proxy is None or
                not hasattr(self.proxy, 'name')
        ):
            Clock.schedule_once(self.finalize, 0)
            return
        if initial:
            self.name = self.proxy.name
            self.paths = self.proxy.setdefault(
                '_image_paths', self.default_image_paths
            )
            zeroes = [0] * len(self.paths)
            self.offxs = self.proxy.setdefault('_offxs', zeroes)
            self.offys = self.proxy.setdefault('_offys', zeroes)
            self.proxy.connect(self._trigger_pull_from_proxy)
        self.bind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys
        )
        self._finalized = True

    def unfinalize(self):
        self.unbind(
            paths=self._trigger_push_image_paths,
            offxs=self._trigger_push_offxs,
            offys=self._trigger_push_offys
        )
        self._finalized = False

    def pull_from_proxy(self, *args):
        initial = not hasattr(self, '_finalized')
        self.unfinalize()
        for key, att in [
                ('_image_paths', 'paths'),
                ('_offxs', 'offxs'),
                ('_offys', 'offys')
        ]:
            if key in self.proxy and self.proxy[key] != getattr(self, att):
                setattr(self, att, self.proxy[key])
        self.finalize(initial)

    def _trigger_pull_from_proxy(self, *args, **kwargs):
        Clock.unschedule(self.pull_from_proxy)
        Clock.schedule_once(self.pull_from_proxy, 0)

    @trigger
    def _trigger_push_image_paths(self, *args):
        self.proxy['_image_paths'] = list(self.paths)

    @trigger
    def _trigger_push_offxs(self, *args):
        self.proxy['_offxs'] = list(self.offxs)

    @trigger
    def _trigger_push_offys(self, *args):
        self.proxy['_offys'] = list(self.offys)

    @trigger
    def _trigger_push_stackhs(self, *args):
        self.proxy['_stackhs'] = list(self.stackhs)

    def on_linecolor(self, *args):
        """If I don't yet have the instructions for drawing the selection box
        in my canvas, put them there. In any case, set the
        :class:`Color` instruction to match my current ``linecolor``.

        """
        if hasattr(self, 'color'):
            self.color.rgba = self.linecolor
            return

        def upd_box_points(*args):
            self.box.points = [
                self.x, self.y,
                self.right, self.y,
                self.right, self.top,
                self.x, self.top,
                self.x, self.y
            ]
        boxgrp = InstructionGroup()
        self.color = Color(*self.linecolor)

        boxgrp.add(self.color)
        self.box = Line()
        upd_box_points()
        self.bind(
            pos=upd_box_points,
            size=upd_box_points
        )
        boxgrp.add(self.box)
        boxgrp.add(Color(1., 1., 1.))
        self.group.add(boxgrp)

    @trigger
    def restack(self, *args):
        childs = sorted(list(self.children), key=lambda child: child.priority, reverse=True)
        self.clear_widgets()
        for child in childs:
            self.add_widget(child)
        self.do_layout()

    def add_widget(self, wid, index=None, canvas=None):
        """Put the widget's canvas in my ``board``'s ``pawnlayout`` rather
        than my own canvas.

        The idea is that all my child widgets are to be instances of
        :class:`Pawn`, and should therefore be drawn after every
        non-:class:`Pawn` widget, so that pawns are on top of spots
        and arrows.

        """
        if index is None:
            for index, child in enumerate(self.children, start=1):
                if wid.priority < child.priority:
                    index = len(self.children) - index
                    break
        super().add_widget(wid, index=index, canvas=canvas)
        self._childs[wid.uid] = wid
        if not hasattr(wid, 'group'):
            return
        wid._no_use_canvas = True
        mycanvas = (
            self.canvas.after if canvas == 'after' else
            self.canvas.before if canvas == 'before' else
            self.canvas
        )
        pawncanvas = (
            self.board.pawnlayout.canvas.after if canvas == 'after' else
            self.board.pawnlayout.canvas.before if canvas == 'before' else
            self.board.pawnlayout.canvas
        )
        mycanvas.remove(wid.canvas)
        for child in self.children:
            if hasattr(child, 'group'):
                if child.group in pawncanvas.children:
                    pawncanvas.remove(child.group)
                pawncanvas.add(child.group)
            else:
                pawncanvas.add(child.canvas)
        self._trigger_layout()

    def remove_widget(self, widget):
        del self._childs[widget.uid]
        super().remove_widget(widget)
        self._trigger_layout()

    def do_layout(self, *args):
        Logger.debug("PawnSpot: {} is laying-out".format(self.name))
        xpad = self.proxy.get('_xpad', 32)
        ypad = self.proxy.get('_ypad', 32)
        self.gutter = gutter = self.proxy.get('_gutter', xpad/2)
        height = self.height - ypad
        content_height = 0
        too_tall = False
        width = self.width - xpad
        content_width = 0
        groups = defaultdict(list)
        for child in self.children:
            group = child.proxy.get('_group', '')
            groups[group].append(child)
            if child.height > height:
                height = child.height
                too_tall = True
        piles = {}
        # Arrange the groups into piles that will fit in me vertically
        for group, members in groups.items():
            members.sort(key=lambda x: x.width * x.height, reverse=True)
            high = 0
            subgroups = []
            subgroup = []
            for member in members:
                high += member.height
                if high > height:
                    subgroups.append(subgroup)
                    subgroup = [member]
                    high = member.height
                else:
                    subgroup.append(member)
            subgroups.append(subgroup)
            content_height = max((content_height, sum(wid.height for wid in subgroups[0])))
            content_width += sum(max(wid.width for wid in subgrp) for subgrp in subgroups)
            piles[group] = subgroups
        self.content_width = content_width + gutter * (len(piles) - 1)
        too_wide = content_width > width
        # If I'm big enough to fit all this stuff, calculate an offset that will ensure
        # it's all centered. Otherwise just offset by my padding so the user can still
        # reach me underneath all the pawns.
        if too_wide:
            offx = xpad
        else:
            offx = self.width / 2 - content_width / 2
        if too_tall:
            offy = ypad
        else:
            offy = self.height / 2 - content_height / 2
        positions = {}
        for pile, subgroups in sorted(piles.items()):
            for subgroup in subgroups:
                subw = subh = 0
                for member in subgroup:
                    subw = max((subw, member.width))
                    positions[member.uid] = (offx, offy + subh)
                    subh += member.height
                offx += subw
            offx += gutter
        self.positions = positions

    def _position(self, *args):
        Logger.debug("PawnSpot: {} repositioning children".format(self.name))
        x, y = self.pos
        for (member_id, (offx, offy)) in self.positions.items():
            self._childs[member_id].pos = self.to_local(x + offx, y + offy)


kv = """
<PawnSpot>:
    linecolor: [0., 1., 1., 1.] if self.selected else [0., 0., 0., 0.]
"""
Builder.load_string(kv)
