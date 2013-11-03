from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.layout import Layout
from kivy.uix.widget import Widget
from kivy.uix.stencilview import StencilView
from kivy.app import App
from kivy.properties import (
    NumericProperty,
    StringProperty,
    ObjectProperty,
    DictProperty,
    ListProperty)


class ThingCalendarColumn(object):
    def update(self, *args):
        if not hasattr(self, 'cells'):
            self.cells = {}
        calendar = self.parent
        thing = calendar.referent
        if self.branch not in thing.locations:
            return
        rditer = thing.locations[self.branch].iterrows()
        try:
            prev = next(rditer)
        except StopIteration:
            return
        done_for = set()
        for rd in rditer:
            if id(rd) not in self.cells:
                cc = self.add_cell(
                    prev["location"], prev["tick_from"], rd["tick_from"])
                self.cells[id(rd)] = cc
            else:
                cc = self.cells[id(rd)]
                cc.text = prev['location']
                cc.tick_from = prev['tick_from']
                cc.tick_to = rd['tick_from']
            done_for.add(id(rd))
            prev = rd
        if None in self.cells:
            indefcc = self.cells[None]
            indefcc.text = prev["location"]
            indefcc.tick_from = prev["tick_from"]
        else:
            indefcc = self.add_cell(prev["location"], prev["tick_from"])
            self.cells[None] = indefcc
        for cell in self.children:
            assert(cell in self.cells.viewvalues())
        undone = set(self.cells.viewkeys()) - done_for - set([None])
        for ccid in undone:
            self.remove_widget(self.cells[ccid])
            del self.cells[ccid]


class Timeline(Widget):
    calendar = ObjectProperty()

    def get_pos(self, branch, tick):
        column = self.calendar.columns[branch]
        x = column.x
        if tick is None:
            y = column.y
        elif tick == 0:
            y = self.calendar.top
        else:
            y = self.calendar.tick_y(tick)
        print("timeline's new coords are {} {}".format(x, y))
        return (x, y)

    def get_line_points(self, x, y):
        return (x, y, x+self.calendar.col_default_width, y)

    def get_wedge_points(self, x, y):
        t = y + 8
        b = y - 8
        return (x, t, x+16, y, x, b)


class BranchConnector(Widget):
    wedge_height = 8

    def on_parent(self, *args):
        self.x = (self.parent.parent_branch_col.window_right -
                  self.parent.parent_branch_col.style.spacing)
        self.y = (self.parent.calendar.tick_to_y(self.column.start_tick) +
                  self.parent.calendar.offy)
        with self.canvas:
            Color(*self.color)
            Line(points=self.get_line_points())
            Triangle(points=self.get_wedge_points())

    def get_line_points(self):
        y0 = self.y
        y2 = y0
        y1 = y0 + self.wedge_height
        x0 = self.x
        # x1 gotta be half way between x0 and the center of my column
        x2 = self.column.left + self.column.width / 2
        dx = x2 - x0
        x1 = x0 + dx
        return [x0, y0, x1, y0, x1, y1, x2, y1, x2, y2]

    def get_wedge_points(self):
        b = self.y
        t = b + self.wedge_height
        c = self.column.left + self.column.width / 2
        rx = self.wedge_width / 2
        l = c - rx
        r = c + rx
        return [c, b, r, t, l, t]


