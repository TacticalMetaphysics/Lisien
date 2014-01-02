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
        self.size = [1, 1]

    def recalc_size(self):
        width = height = 1
        for texture in self.texs:
            if texture.width > width:
                width = texture.width
            if texture.height > height:
                height = texture.height
        self.size = (width, height)

    def rectify(self, tex):
        rect = Rectangle(
            pos=self.pos,
            size=tex.size,
            texture=tex)
        self.texture_rectangles[tex] = rect
        group = InstructionGroup()
        group.add(rect)
        self.rectangle_groups[rect] = group
        return group

    def insert(self, i, tex):
        self.suppressor = True
        if not self.canvas:
            Clock.schedule_once(
                lambda dt: self.insert(i, tex), 0)
            return
        self.texs.insert(i, tex)
        group = self.rectify(tex)
        self.canvas.insert(i, group)
        if tex.width > self.width:
            self.width = tex.width
        if tex.height > self.height:
            self.height = tex.height
        self.suppressor = False

    def append(self, tex):
        self.suppressor = True
        self.texs.append(tex)
        group = self.rectify(tex)
        self.canvas.add(group)
        if tex.width > self.width:
            self.width = tex.width
        if tex.height > self.height:
            self.height = tex.height
        self.suppressor = False

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
        self.recalc_size()
        self.suppressor = False

    def __setitem__(self, i, v):
        self.__delitem__(i)
        self.insert(i, v)

    def pop(self, i=-1):
        tex = self[i]
        del self[i]
        return tex

    def on_pos(self, *args):
        for rectangle in self.rectangle_groups.iterkeys():
            rectangle.pos = self.pos
