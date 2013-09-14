from pyglet.graphics import OrderedGroup
from pyglet.image import SolidColorImagePattern
from pyglet.sprite import Sprite
from time import time


class PicPanel:
    """Icon for a particular picture in a PicPicker."""
    atrdic = {
        "right": lambda self: self.left + self.pic.width,
        "bot": lambda self: self.left + self.pic.width,
        "window_left": lambda self: self.picker.window_left + self.left,
        "window_top": lambda self: self.picker.window_top - self.top + self.picker.scrolled_px,
        "window_bot": lambda self: self.picker.window_top - self.bot + self.picker.scrolled_px,
        "window_right": lambda self: self.picker.window_left + self.pic.width,
        "tex": lambda self: self.pic.tex,
        "texture": lambda self: self.pic.tex,
        "width": lambda self: self.pic.width,
        "height": lambda self: self.pic.height,
        "pressed": lambda self: self.window.pressed is self,
        "hovered": lambda self: self.window.hovered is self,
        "in_picker": lambda self: (self.window_top > self.picker.window_bot and
                              self.window_bot < self.picker.window_top)}

    def __init__(self, picker, pic):
        self.picker = picker
        self.window = self.picker.window
        self.closet = self.picker.closet
        self.pic = pic
        self.sprite = None
        self.tweaks = 0

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
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

    def get_state_tup(self):
        return (
            self.pic,
            self.left,
            self.right,
            self.top,
            self.bot,
            self.tweaks)

    def draw(self, batch, group):
        pickerfggroup = OrderedGroup(1, group)
        if self.in_picker:
            try:
                self.sprite.x = self.window_left
                self.sprite.y = self.window_bot
            except:
                self.sprite = Sprite(
                    self.tex,
                    self.window_left,
                    self.window_bot,
                    batch=batch,
                    group=group)
        else:
            self.delete()


class PicPicker:
    """For picking pictures.

Parameter targetn is the name of a window attribute. The pic picked
will be assigned to that attribute of the window the picker is in.

    """
    atrdic = {
        "window_left": lambda self: int(self.left_prop * self.window.width),
        "window_right": lambda self: int(self.right_prop * self.window.width),
        "window_top": lambda self: int(self.top_prop * self.window.height),
        "window_bot": lambda self: int(self.bot_prop * self.window.height),
        "width": lambda self: self.window_right - self.window_left,
        "height": lambda self: self.window_top - self.window_bot,
        "imgs": lambda self: self.closet.imgdict.itervalues(),
        "hovered": lambda self: self is self.window.hovered,
        "pressed": lambda self: self is self.window.pressed,
        "bgpat": lambda self: {
            True: self.bgpat_active,
            False: self.bgpat_inactive}[self.hovered],
        "on_screen": lambda self: (self.window_top > 0 and
                              self.window_bot < self.window.height and
                              self.window_right > 0 and
                              self.window_left < self.window.width)}

    def __init__(self, window, left, top, bot, right, style, targetn, flagn):
        self.window = window
        self.closet = self.window.closet
        self.left_prop = left
        self.top_prop = top
        self.right_prop = right
        self.bot_prop = bot
        self.style = style
        self.targetn = targetn
        self.flagn = flagn
        self.pixrows = []
        self.scrolled_to_row = 0
        self.scrolled_px = 0
        self.tweaks = 0
        self.oldstate = None
        self.sprite = None
        self.bgpat_inactive = SolidColorImagePattern(style.bg_inactive.tup)
        self.bgimg = self.bgpat_inactive.create_image(self.width, self.height)
        self.bgpat_active = SolidColorImagePattern(style.bg_active.tup)
        self.panels = [PicPanel(self, img) for img in self.imgs]

    def __getattr__(self, attrn):
        try:
            return self.atrdic[attrn](self)
        except KeyError:
            raise AttributeError(
                "PicPicker instance has no attribute named " + attrn)

    def layout(self):
        new_pixrows = []
        panels = list(self.panels)
        nexttop = self.style.spacing
        nextleft = self.style.spacing
        rightmargin = self.window_right
        while panels != []:
            row = []
            rowheight = 0
            while panels != [] and nextleft < rightmargin:
                panel = panels.pop()
                if panel.width > self.width:
                    continue
                panel.left = nextleft
                nextleft = panel.right + self.style.spacing
                if nextleft > rightmargin:
                    panels.insert(0, panel)
                    break
                panel.top = nexttop
                if panel.height > rowheight:
                    rowheight = panel.height
                row.append(panel)
            nexttop += rowheight + self.style.spacing
            nextleft = self.style.spacing
            new_pixrows.append(row)
        self.pixrows = new_pixrows

    def rowhash(self):
        rowshashes = [hash(tuple(row)) for row in self.pixrows]
        return hash(tuple(rowshashes))

    def get_state_tup(self):
        return (
            self.rowhash(),
            self.style,
            self.window_left,
            self.window_right,
            self.window_top,
            self.window_bot,
            self.tweaks)

    def overlaps(self, x, y):
        return (
            x > self.window_left and
            x < self.window_right and
            y > self.window_bot and
            y < self.window_top)

    def hover(self, x, y):
        # Relativize the coordinates to my top left corner. That means
        # y gets lower on the screen as it ascends.
        x -= self.window_left
        y = self.window_top - y
        # Iterate thru my pix and see if one overlaps. If so, return it
        for row in self.pixrows:
            for panel in row:
                if (
                        panel.left < x and
                        panel.right > x and
                        panel.top < y and
                        panel.bot > y):
                    return panel
        # If not, return myself
        return self

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass
        for row in self.pixrows:
            for pic in row:
                pic.delete()

    def scroll_down_once(self):
        if self.scrolled_to_row + 1 == len(self.pixrows):
            return
        rowheight = max([
            pic.height for pic in
            self.pixrows[self.scrolled_to_row]])
        self.scrolled_px += rowheight
        self.scrolled_to_row += 1

    def scroll_up_once(self):
        if self.scrolled_to_row == 0:
            return
        rowheight = max([
            pic.height for pic in
            self.pixrows[self.scrolled_to_row]])
        self.scrolled_px -= rowheight
        self.scrolled_to_row -= 1

    def draw(self):
        self.sprite = Sprite(
            self.bgimg,
            self.window_left,
            self.window_bot,
            batch=self.window.batch,
            group=self.window.menu_bg_group)
        self.layout()
        for pixrow in self.pixrows:
            for pic in pixrow:
                pic.draw(self.window.batch, self.window.menu_bg_group)
