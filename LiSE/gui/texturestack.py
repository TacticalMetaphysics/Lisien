"""A widget with a list-like interface. Give it textures, and it will
draw those textures in itself."""
from kivy.uix.widget import Widget
from kivy.graphics import (
    Rectangle,
    InstructionGroup
)
from kivy.properties import (
    ListProperty,
    DictProperty)


class TextureStack(Widget):
    texs = ListProperty([])
    texture_rectangles = DictProperty({})
    rectangle_groups = DictProperty({})

    def __init__(self, **kwargs):
        super(TextureStack, self).__init__(**kwargs)
        self.big_instruction = InstructionGroup()
        self.canvas.add(self.big_instruction)

    def clear(self):
        self.big_instruction.clear()
        self.rectangle_groups = {}
        self.texture_rectangles = {}
        self.texs = []

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
        self.texs.insert(i, tex)
        group = self.rectify(tex)
        self.big_instruction.insert(i, group)

    def append(self, tex):
        self.texs.append(tex)
        group = self.rectify(tex)
        self.big_instruction.add(group)

    def remove_tex(self, tex):
        self.texs.remove(tex)
        self.remove_rect(self.texture_rectangles[tex])
        del self.texture_rectangles[tex]

    def remove_rect(self, rect):
        group = self.rectangle_groups[rect]
        self.big_instruction.remove(group)
        del self.rectangle_groups[rect]

    def remove(self, tex_rect):
        if isinstance(tex_rect, Rectangle):
            self.remove_rect(tex_rect)
        else:
            self.remove_tex(tex_rect)

    def __delitem__(self, i):
        tex = self.texs[i]
        rect = self.texture_rectangles[tex]
        group = self.rectangle_groups[rect]
        self.big_instruction.remove(group)
        del self.rectangle_groups[rect]
        del self.texture_rectangles[tex]
        del self.texs[i]

    def __setitem__(self, i, v):
        self.__delitem__(i)
        self.insert(i, v)

    def pop(self, i=-1):
        tex = self[i]
        del self[i]
        return tex
