from util import SaveableMetaclass
from pyglet.image import SolidColorImagePattern


__metaclass__ = SaveableMetaclass


class PicPicker:
    """For picking pictures."""
    def __init__(self, window, left, top, right, bot, style):
        self.window = window
        self.db = self.window.db
        self.left_prop = left
        self.top_prop = top
        self.right_prop = right
        self.bot_prop = bot
        self.style = style
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
        elif attrn == 'imgs':
            # TODO some way to filter images, like by name or whatever
            return self.db.imgdict.itervalues()
        else:
            raise AttributeError(
                "PicPicker instance has no attribute named " + attrn)
