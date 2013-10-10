from kivy.graphics import Color, InstructionGroup
from kivy.uix.image import Image


class PicPanel(Image):
    """Icon for a particular picture in a PicPicker."""
    atrdic = {
        "right": lambda self: self.left + self.pic.width,
        "bot": lambda self: self.left + self.pic.width,
        "window_left": lambda self: self.picker.window_left + self.left,
        "window_top": lambda self:
        self.picker.window_top - self.top + self.picker.scrolled_px,
        "window_bot": lambda self:
        self.picker.window_top - self.bot + self.picker.scrolled_px,
        "window_right": lambda self:
        self.picker.window_left + self.pic.width,
        "tex": lambda self: self.pic.tex,
        "texture": lambda self: self.pic.tex,
        "width": lambda self: self.pic.width,
        "height": lambda self: self.pic.height,
        "pressed": lambda self: self.window.pressed is self,
        "hovered": lambda self: self.window.hovered is self,
        "in_picker": lambda self:
        (self.window_top > self.picker.window_bot and
         self.window_bot < self.picker.window_top)}

    def __init__(self, picker, pic):
        self.picker = picker
        self.window = self.picker.window
        self.closet = self.picker.closet
        Image.__init__(self, pic.texture)

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn]()
        except KeyError:
            raise AttributeError(
                "PicPanel instance has no attribute named " + attrn)

    def __hash__(self):
        return hash(self.get_state_tup())

    def __str__(self):
        return str(self.pic)

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass

    def onclick(self, x, y, button, modifiers):
        self.picker.delete()
        setattr(self.window, self.picker.targetn, self.pic)
        setattr(self.window, self.picker.flagn, True)

    def overlaps(self, x, y):
        x -= self.picker.window_left
        y = self.picker.window_top - y
        return (
            x > self.left and
            x < self.right and
            y > self.top and
            y < self.bot)

    def pass_focus(self):
        return self.picker
