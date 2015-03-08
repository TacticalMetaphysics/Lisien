from multiprocessing import Process
from multiprocessing.managers import BaseManager, BaseProxy
from LiSE.core import Engine


class EngineHandle(object):
    def __init__(self, worlddb, codedb):
        self._real = Engine(worlddb, codedb)

    def commit(self):
        self._real.commit()

    def close(self):
        self._real.close()

    def get_string(self, k):
        return self._real.string[k]

    def set_string(self, k, v):
        self._real.string[k] = v

    def del_string(self, k):
        del self._real.string[k]

    def get_eternal(self, k):
        return self._real.eternal[k]

    def set_eternal(self, k, v):
        self._real.eternal[k] = v

    def del_eternal(self, k):
        del self._real.eternal[k]

    def init_character(self, char, statdict={}):
        if char in self._real.character:
            raise KeyError("Already have character {}".format(char))
        self._real.character[char] = {}
        self._real.character[char].stat.update(statdict)

    def del_character(self, char):
        del self._real.character[char]

    def get_character_stat(self, char, k):
        return self._real.character[char].stat[k]

    def set_character_stat(self, char, k, v):
        self._real.character[char].stat[k] = v

    def del_character_stat(self, char, k):
        del self._real.character[char].stat[k]

    def get_node_stat(self, char, node, k):
        return self._real.character[char].node[node][k]

    def set_node_stat(self, char, node, k, v):
        self._real.character[char].node[node][k] = v

    def del_node_stat(self, char, node, k):
        del self._real.character[char].node[node][k]

    def del_node(self, char, node):
        del self._real.character[char].node[node]

    def init_thing(self, char, thing, statdict={}):
        if thing in self._real.character[char].thing:
            raise KeyError(
                'Already have thing in character {}: {}'.format(
                    char, thing
                )
            )
        self._real.character[char].thing[thing] = statdict

    def get_thing_location(self, char, th):
        return self._real.character[char].thing[th]['location']

    def get_thing_next_location(self, char, th):
        return self._real.character[char].thing[th]['next_location']

    def init_place(self, char, place, statdict={}):
        if place in self._real.character[char].place:
            raise KeyError(
                'Already have place in character {}: {}'.format(
                    char, place
                )
            )
        self._real.character[char].place[place] = statdict

    def init_portal(self, char, o, d, statdict={}):
        if (
                o in self._real.character[char].portal and
                d in self._real.character[char].portal[o]
        ):
            raise KeyError(
                'Already have portal in character {}: {}->{}'.format(
                    char, o, d
                )
            )
        self._real.character[char].portal[o][d] = statdict

    def del_portal(self, char, o, d):
        del self._real.character[char].portal[o][d]

    def get_portal_stat(self, char, o, d, k):
        return self._real.character[char].portal[o][d][k]

    def set_portal_stat(self, char, o, d, k, v):
        self._real.character[char].portal[o][d][k] = v

    def del_portal_stat(self, char, o, d, k):
        del self._real.character[char][o][d][k]


class EngineHandleProxy(BaseProxy):
    def commit(self):
        self._callmethod('commit')

    def close(self):
        self._callmethod('close')

    def get_string(self, k):
        return self._callmethod('get_string', (k,))

    def set_string(self, k, v):
        self._callmethod('set_string', (k, v,))

    def del_string(self, k):
        self._callmethod('del_string', (k,))

    def get_eternal(self, k):
        return self._callmethod('get_eternal', (k,))

    def set_eternal(self, k, v):
        self._callmethod('set_eternal', (k, v,))

    def del_eternal(self, k):
        self._callmethod('del_eternal', (k,))

    def init_character(self, char, statdict={}):
        self._callmethod('init_character', (char, statdict))

    def del_character(self, char):
        self._callmethod('del_character', (char,))

    def get_character_stat(self, char, k):
        return self._callmethod('get_character_stat', (char, k))

    def set_character_stat(self, char, k, v):
        self._callmethod('set_character_stat', (char, k, v))

    def del_character_stat(self, char, k):
        self._callmethod('del_character_stat', (char, k))

    def get_node_stat(self, char, node, k):
        return self._callmethod('get_node_stat', (char, node, k))

    def set_node_stat(self, char, node, k, v):
        self._callmethod('set_node_stat', (char, node, k, v))

    def del_node_stat(self, char, node, k):
        self._callmethod('del_node_stat', (char, node, k))

    def del_node(self, char, node):
        self._callmethod('del_node', (char, node))

    def init_thing(self, char, thing, statdict={}):
        self._callmethod('init_thing', (char, thing, statdict))

    def get_thing_location(self, char, thing):
        return self._callmethod('get_thing_location', (char, thing))

    def get_thing_next_location(self, char, thing):
        return self._callmethod('get_thing_next_location', (char, thing))

    def init_place(self, char, place, statdict={}):
        self._callmethod('init_place', (char, place, statdict))

    def init_portal(self, char, o, d, statdict={}):
        self._callmethod('init_portal', (char, o, d, statdict))

    def del_portal(self, char, o, d):
        self._callmethod('del_portal', (char, o, d))

    def get_portal_stat(self, char, o, d, k):
        return self._callmethod('get_portal_stat', (char, o, d, k))

    def set_portal_stat(self, char, o, d, k, v):
        self._callmethod('set_portal_stat', (char, o, d, k, v))

    def del_portal_stat(self, char, o, d, k):
        self._callmethod('del_portal_stat', (char, o, d, k))


class EngineManager(BaseManager):
    pass


EngineManager.register(
    'EngineHandle', EngineHandle, EngineHandleProxy
)


def test_eng_handle(engine):
    engine.init_character('FooChar', {'nice': True})
    assert(engine.get_character_stat('FooChar', 'nice') is True)
    engine.set_character_stat('FooChar', 'boring', False)
    assert(engine.get_character_stat('FooChar', 'boring') is False)
    engine.close()


if __name__ == "__main__":
    from examples.utiltest import clear_off
    clear_off()
    with EngineManager() as manager:
        engine = manager.EngineHandle('LiSEworld.db', 'LiSEcode.db')
        p = Process(target=test_eng_handle, args=(engine,))
        p.start()
        p.join()
