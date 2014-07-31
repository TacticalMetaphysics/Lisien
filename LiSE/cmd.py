# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) 2013-2014 Zachary Spector,  zacharyspector@gmail.com
"""A command processor, giving access to the engine through a pipe."""
import threading
from collections import deque


commands = deque()
outputs = deque()


def emit(engine, out_pipe, lock):
    while True:
        if commands:
            lock.acquire()  # not sure I even need a lock
            outputs.append(commands.pop(0)())
            lock.release()
        else:
            out_pipe.put(outputs.pop(0))


def receive(engine, in_pipe, lock):
    def setitem(mapping, item, value):
        mapping[item] = value

    def delitem(mapping, item):
        del mapping[item]

    def character_portals(char):
        for o in char.portal:
            for d in char.portal[0]:
                yield (o, d)

    processors = {
        "set_branch": lambda b: setattr(engine, 'branch', b),
        "set_tick": lambda t: setattr(engine, 'tick', t),
        "next_tick": engine.next_tick,
        "add_character": engine.add_character,
        "get_character": lambda name: engine.character[name].copy(),
        "ls_things": lambda name: list(engine.character[name].thing.keys()),
        "ls_places": lambda name: list(engine.character[name].place.keys()),
        "ls_portals": lambda name: list(character_portals(engine.character[name])),
        "del_character": engine.del_character,
        "add_thing": lambda char, th: engine.character[char].add_thing(th),
        "get_thing": lambda char, th: engine.character[char].thing[th].copy(),
        "del_thing": lambda char, th: engine.character[char].del_thing(th),
        "get_thing_loc": lambda char, th: getattr(engine.character[char].thing[th], 'location').copy(),
        "get_thing_conts": lambda char, th: [thing.copy() for thing in engine.character[char].thing[th].contents],
        "ls_thing_stats": lambda char, th: list(engine.character[char].thing[th].keys()),
        "get_thing_stat": lambda char, th, stat: engine.character[char].thing[th][stat],
        "set_thing_stat": lambda char, th, stat, val: setitem(engine.character[char].thing[th], stat, val),
        "del_thing_stat": lambda char, th, stat: delitem(engine.character[char].thing[th], stat),
        "add_place": lambda char, pl: engine.character[char].add_place(pl),
        "get_place": lambda char, pl: engine.character[char].place[pl],
        "del_place": lambda char, pl: delitem(engine.character[char].place, pl),
        "get_place_conts": lambda char, pl: [thing.copy() for thing in engine.character[char].place[pl].contents],
        "ls_place_stats": lambda char, pl: list(engine.character[char].place[pl].keys()),
        "get_place_stat": lambda char, pl, stat: engine.character[char].place[pl][stat],
        "set_place_stat": lambda char, pl, stat, val: setitem(engine.character[char].place[pl], stat, val),
        "del_place_stat": lambda char, pl, stat: delitem(engine.character[char].place[pl], stat),
        "add_portal": lambda char, o, d: engine.character[char].add_portal(o, d),
        "add_portal_symm": lambda char, o, d: engine.character[char].add_portal(o, d, symmetrical=True),
        "get_portal": lambda char, o, d: engine.character[char].portal[o][d].copy(),
        "del_portal": lambda char, o, d: delitem(engine.character[char].portal[o], d),
        "get_portal_conts": lambda char, o, d: [thing.copy() for thing in engine.character[char].portal[o][d].contents],
        "ls_portal_stats": lambda char, o, d: list(engine.character[char].portal[o][d].keys()),
        "get_portal_stat": lambda char, o, d, stat: engine.character[char].portal[o][d][stat],
        "set_portal_stat": lambda char, o, d, stat, val: setitem(
            engine.character[char].portal[o][d], stat, val
        ),
        "del_portal_stat": lambda char, o, d, stat: delitem(engine.character[char].portal[o][d], stat),
        }
    def process(cmd, args):
        try:
            return (cmd, processors[cmd](*args))
        except Exception as ex:
            return (cmd, ex)
    while True:
        tup = in_pipe.get()
        cmd = tup[0]
        args = tup[1:]
        commands.append(lambda: process(cmd, args))


def connect(engine, in_pipe, out_pipe):
    lock = threading.Lock()
    emitter = threading.Thread(target=emit, args=(engine, out_pipe, lock))
    receiver = threading.Thread(target=receive, args=(engine, in_pipe, lock))
    emitter.start()
    receiver.start()
