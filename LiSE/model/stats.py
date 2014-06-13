from collections import MutableMapping

class Stats(MutableMapping):
    def __init__(self, closet, keys, mkbone):
        self.closet = closet
        self.keys = keys
        self.mkbone = mkbone

    def __contains__(self, that):
        skel_keys = list(self.keys)
        skel = self.closet.skeleton[skel_keys.pop(0)]
        while skel_keys:
            skel = skel[skel_keys.pop(0)]
        return that in skel

    def __getitem__(self, key):
        def get_stat_recursive(branch, tick, skel, key):
            if key in skel and branch in skel[key]:
                bone = skel[key][branch].value_during(tick)
                if bone.value is None:
                    raise KeyError(
                        "{} is not applicable right now".format(key)
                    )
                typ = {
                    'boolean': bool,
                    'integer': int,
                    'real': float,
                    'text': unicode
                }[bone.type]
                return typ(bone.value)
            elif branch == 0:
                raise KeyError(
                    "{} is *never* applicable".format(key)
                )
            else:
                return get_stat_recursive(
                    self.closet.timestream.parent(branch),
                    tick,
                    skel,
                    key
                )

        (branch, tick) = self.closet.timestream.time
        skel_keys = list(self.keys)
        skel = self.closet.skeleton[skel_keys.pop(0)]
        while skel_keys:
            skel = skel[skel_keys.pop(0)]
        return get_stat_recursive(branch, tick, skel, key)

    def __setitem__(self, key, value):
        (branch, tick) = self.closet.timestream.time
        self.closet.set_bone(self.mkbone(branch, tick, key, value))

    def __delitem__(self, key):
        self[key] = None
