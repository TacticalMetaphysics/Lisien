class Contents(set):
    def __init__(self, character, name):
        self.character = character
        self.name = name

    def __repr__(self):
        return str([thing for thing in self])

    def add(self, that):
        that['location'] = self.name

    def remove(self, that):
        if that['location'] != self.name:
            raise KeyError("{} not in {}".format(that.name, self.name))
        self.discard(that)

    def discard(self, that):
        that['location'] = None

    def __iter__(self):
        (branch, tick) = self.closet.timestream.time
        skel = (
            self.character.closet.skeleton
            ['thing_stat']
            [self.character.name]
        )
        for thing in skel:
            if (
                    'location' in skel[thing] and
                    branch in skel[thing]['location'] and
                    skel[thing]['location']
                    [branch].value_during(tick).value == self.name
            ):
                yield thing

    def __contains__(self, that):
        return that['location'] == self.name
