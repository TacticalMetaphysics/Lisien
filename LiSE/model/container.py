from LiSE.orm import SaveableMetaclass


class Container(object):
    __metaclass__ = SaveableMetaclass

    def __contains__(self, it):
        return it.location is self

    def iter_contents(self, branch=None, tick=None):
        (branch, tick) = self.character.sanetime(branch, tick)
        skel = self.character.closet.skeleton[u"thing"][
            unicode(self.character)]
        for thingbone in skel.iterbones():
            if thingbone.host == unicode(self.character):
                locbone = self.character.closet.skeleton[u"thing_loc"][
                    thingbone.character][thingbone.name][
                    branch].value_during(tick)
                if locbone.location == unicode(self):
                    char = self.character.closet.get_character(
                        thingbone.character)
                    yield char.get_thing(thingbone.name)

    def get_contents(self, branch=None, tick=None):
        return set(self.iter_contents(branch, tick))

    def subjective_lookup(self, method, observer=None, xargs=[]):
        if observer is None:
            callabl = getattr(self.character, method)
        else:
            facade = self.character.get_facade(observer)
            callabl = getattr(facade, method)
        args = [self.name] + xargs
        return callabl(*args)
