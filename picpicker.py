from pyglet.image import SolidColorImagePattern
from time import time

class PicPanel:
    """Icon for a particular picture in a PicPicker."""
    def __init__(self, picker, pic):
        self.picker = picker
        self.window = self.picker.window
        self.rumor = self.picker.rumor
        self.pic = pic
        self.sprite = None
        self.tweaks = 0

    def __getattr__(self, attrn):
        if attrn == "right":
            return self.left + self.pic.width
        elif attrn == "bot":
            return self.top + self.pic.height
        elif attrn == "window_left":
            return self.picker.window_left + self.left
        elif attrn == "window_top":
            return self.picker.window_top - self.top + self.picker.scrolled_px
        elif attrn == "window_bot":
            return self.picker.window_top - self.bot + self.picker.scrolled_px
        elif attrn == "window_right":
            return self.picker.window_left + self.pic.width
        elif attrn in ("tex", "texture"):
            return self.pic.tex
        elif attrn == 'width':
            return self.pic.width
        elif attrn == 'height':
            return self.pic.height
        elif attrn == "pressed":
            return self.window.pressed is self
        elif attrn == "hovered":
            return self.window.hovered is self
        elif attrn == "in_picker":
            return (
                self.window_top > self.picker.window_bot and
                self.window_bot < self.picker.window_top)
        else:
            raise AttributeError(
                "PicPanel instance has no attribute named " + attrn)

    def __hash__(self):
        return hash(self.get_state_tup())

    def __str__(self):
        return str(self.pic)

    def delete(self):
        print "{0}: PicPanel showing {1} deleted".format(time(), str(self.pic))
        try:
            self.sprite.delete()
        except:
            pass

    def onclick(self):
        print "{0}: PicPanel showing {1} received click".format(time(), str(self.pic))
        self.picker.delete()
        setattr(self.window, self.picker.targetn, self.pic)
        self.window.set_mouse_cursor_texture(self.tex)

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


class PicPicker:
    """For picking pictures.

Parameter targetn is the name of a window attribute. The pic picked
will be assigned to that attribute of the window the picker is in.

    """
    def __init__(self, window, left, top, bot, right, style, targetn):
        self.window = window
        self.rumor = self.window.rumor
        self.left_prop = left
        self.top_prop = top
        self.right_prop = right
        self.bot_prop = bot
        self.style = style
        self.targetn = targetn
        self.pixrows = []
        self.scrolled_to_row = 0
        self.scrolled_px = 0
        self.tweaks = 0
        self.oldstate = None
        self.sprite = None
        self.bgpat_inactive = SolidColorImagePattern(style.bg_inactive.tup)
        self.bgpat_active = SolidColorImagePattern(style.bg_active.tup)
        self.panels = [PicPanel(self, img) for img in self.imgs]
        print "{0}: Instantiated a PicPicker targeting {1}".format(time(), targetn)

    def __getattr__(self, attrn):
        if attrn == 'window_left':
            return int(self.left_prop * self.window.width)
        elif attrn == 'window_right':
            return int(self.right_prop * self.window.width)
        elif attrn == 'window_top':
            return int(self.top_prop * self.window.height)
        elif attrn == 'window_bot':
            return int(self.bot_prop * self.window.height)
        elif attrn == 'width':
            return self.window_right - self.window_left
        elif attrn == 'height':
            return self.window_top - self.window_bot
        elif attrn == 'imgs':
            # TODO some way to filter images, like by name or whatever
            return self.rumor.imgdict.itervalues()
        elif attrn == 'hovered':
            return self is self.window.hovered
        elif attrn == 'pressed':
            return self is self.window.pressed
        elif attrn == 'bgpat':
            if self.hovered:
                return self.bgpat_active
            else:
                return self.bgpat_inactive
        elif attrn == 'on_screen':
            return (
                self.window_top > 0 and
                self.window_bot < self.window.height and
                self.window_right > 0 and
                self.window_left < self.window.width)
        else:
            raise AttributeError(
                "PicPicker instance has no attribute named " + attrn)

    def layout(self):
        print "{0}: Laying out PicPicker targeting {1}".format(time(), self.targetn)
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

    def hovered(self, x, y):
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
        print "{0}: PicPicker scrolled down".format(time())
        if self.scrolled_to_row + 1 == len(self.pixrows):
            return
        rowheight = max([
            pic.height for pic in
            self.pixrows[self.scrolled_to_row]])
        self.scrolled_px += rowheight
        self.scrolled_to_row += 1

    def scroll_up_once(self):
        print "{0}: PicPicker scrolled up".format(time())
        if self.scrolled_to_row == 0:
            return
        rowheight = max([
            pic.height for pic in
            self.pixrows[self.scrolled_to_row]])
        self.scrolled_px -= rowheight
        self.scrolled_to_row -= 1
