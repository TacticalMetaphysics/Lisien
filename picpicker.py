from pyglet.image import SolidColorImagePattern

class PicPanel:
    """Icon for a particular picture in a PicPicker."""
    def __init__(self, picker, pic, left, top):
        self.picker = self.picker
        self.rumor = self.picker.rumor
        self.pic = pic
        self.left = left
        self.top = top
        self.sprite = None

    def __getattr__(self, attrn):
        if attrn == "window_left":
            return self.picker.window_left + self.left
        elif attrn == "window_top":
            return self.picker.window_top - self.top
        elif attrn == "window_bot":
            return self.picker.window_top - self.bot
        elif attrn == "window_right":
            return self.picker.window_left + self.right
        elif attrn in ("tex", "texture"):
            return self.pic.tex
        else:
            return getattr(self.pic, attrn)

    def __hash__(self):
        return hash(self.pic)

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass


class PicPicker:
    """For picking pictures."""
    def __init__(self, window, left, top, right, bot, style):
        self.window = window
        self.rumor = self.window.rumor
        self.left_prop = left
        self.top_prop = top
        self.right_prop = right
        self.bot_prop = bot
        self.style = style
        self.visible = False
        self.oldstate = None
        self.sprite = None
        self.bgpat_inactive = SolidColorImagePattern(style.bg_inactive.tup)
        self.bgpat_active = SolidColorImagePattern(style.bg_active.tup)

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
            return self.db.imgdict.itervalues()
        elif attrn == 'hovered':
            return self is self.window.hovered
        else:
            raise AttributeError(
                "PicPicker instance has no attribute named " + attrn)

    def layout(self):
        for row in iter(self.pixrows):
            for panel in iter(row):
                panel.delete()
        self.pixrows = []
        while True:
            try:
                imgiter = self.imgs
                nextleft = self.style.spacing
                nexttop = self.style.spacing
                row = []
                while nextleft < self.width:
                    img = imgiter.next()
                    pp = PicPanel(self, img, nextleft, nexttop)
                    row.append(pp)
                    nextleft = pp.right + self.style.spacing
                maxheight = max([panl.height for panl in row])
                for panl in row:
                    panl.bot = panl.top + maxheight
                    panl.right = panl.left + panl.width
                self.pixrows.append(row)
                nexttop = pr.bot_from_top + self.style.spacing
            except StopIteration:
                return
            
    def rowhash(self):
        rowshashes = [hash(tuple(row)) for row in self.pixrows]
        return hash(tuple(rowshashes))

    def get_state_tup(self):
        return (
            self.rowhash(),
            self.visible,
            self.style,
            self.window_left,
            self.window_right,
            self.window_top,
            self.window_bot)

    def delete(self):
        try:
            self.sprite.delete()
        except:
            pass
        for row in self.pixrows:
            for pic in row:
                pic.delete()
