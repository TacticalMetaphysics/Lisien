from LiSE.orm import SaveableMetaclass


class Container(object):
    __metaclass__ = SaveableMetaclass

    def __contains__(self, it):
        return it.location is self

    def iter_contents(self, branch=None, tick=None):
        skel = self.character.closet.skeleton[u"thing"][
            unicode(self.character)
        ]
        for thingbone in skel.iterbones():
            if thingbone.host == unicode(self.character):
                locbone = self.character.closet.get_timely(
                    [u"thing_loc", thingbone.character, thingbone.name],
                    branch,
                    tick
                )
                if locbone.location == unicode(self):
                    char = self.character.closet.get_character(
                        thingbone.character
                    )
                    yield char.get_thing(thingbone.name)

    def get_contents(self, branch=None, tick=None):
        return set(self.iter_contents(branch, tick))
