"""A widget with a list-like interface. Give it textures, and it will
draw those textures in itself."""
from kivy.uix.widget import Widget
from kivy.graphics import (
    Rectangle,
    InstructionGroup
)
from kivy.properties import (
    BooleanProperty,
    ListProperty,
    DictProperty)
from kivy.clock import Clock


class TextureStack(Widget):
    texs = ListProperty([])
    offxs = ListProperty([])
    texture_rectangles = DictProperty({})
    rectangle_groups = DictProperty({})
    suppressor = BooleanProperty(False)

    def __init__(self, **kwargs):
        super(TextureStack, self).__init__(**kwargs)
        if len(self.texs) == 0:
            self.size = [1, 1]
        else:
            self.recalc_size()

    def on_texs(self, *args):
        if self.suppressor:
            return
        texs = list(self.texs)
        while texs != []:
            tex = texs.pop()
            if tex not in self.texture_rectangles:
                self.suppressor = True
                self[len(texs)] = tex
                self.suppressor = False

    def clear(self):
        self.canvas.clear()
        self.rectangle_groups = {}
        self.texture_rectangles = {}
        self.texs = []
        self.offxs = []
        self.offys = []
        self.size = [1, 1]

    def recalc_size(self):
        width = height = 1
        for i in xrange(0, len(self.texs)):
            width = max([self.texs[i].width + self.offxs[i], width])
            height = max([self.texs[i].height + self.offys[i], height])
        self.size = (width, height)

    def rectify(self, tex, offx=0, offy=0):
        if offx < 0:
            self.offxs = map(lambda x: x-offx, self.offxs)
            offx = 0
        if offy < 0:
            self.offys = map(lambda y: y-offy, self.offys)
            offy = 0
        rect = Rectangle(
            x=self.x+offx,
            y=self.y+offy,
            pos=self.pos,
            size=tex.size,
            texture=tex)
        self.texture_rectangles[tex] = rect
        group = InstructionGroup()
        group.add(rect)
        self.rectangle_groups[rect] = group
        return group

    def insert(self, i, tex, offx=0, offy=0):
        self.suppressor = True
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: self.insert(i, tex, offx, offy), 0)
            return
        self.texs.insert(i, tex)
        self.offxs.insert(i, offx)
        self.offys.insert(i, offy)
        group = self.rectify(tex, offx, offy)
        self.canvas.insert(i, group)
        self.width = max([self.width, tex.width + max([offx, 0])])
        self.height = max([self.height, tex.height + max([offy, 0])])
        self.suppressor = False

    def append(self, tex, offx=0, offy=0):
        self.insert(len(self.texs), tex, offx, offy)

    def __delitem__(self, i):
        self.suppressor = True
        tex = self.texs[i]
        try:
            rect = self.texture_rectangles[tex]
            group = self.rectangle_groups[rect]
            self.canvas.remove(group)
            del self.rectangle_groups[rect]
            del self.texture_rectangles[tex]
        except KeyError:
            pass
        del self.texs[i]
        del self.offxs[i]
        del self.offys[i]
        self.recalc_size()
        self.suppressor = False

    def __setitem__(self, i, v, offx=0, offy=0):
        self.__delitem__(i)
        self.insert(i, v, offx, offy)

    def pop(self, i=-1):
        self.suppressor = True
        tex = self.texs.pop(i)
        self.offxs.pop(i)
        self.offys.pop(i)
        self.suppressor = False
        self.on_texs()
        return tex

    def on_pos(self, *args):
        for i in xrange(0, len(self.texs)):
            tex = self.texs[i]
            offx = self.offxs[i]
            offy = self.offys[i]
            rect = self.texture_rectangles[tex]
            rect.pos = (self.x + offx, self.y + offy)
