import unittest
import re
from functools import reduce
from collections import defaultdict
from allegedb.cache import StructuredDefaultDict, WindowDict
from LiSE.engine import Engine
from LiSE.examples import college as sim


def deepDictDiffIter(d0, d1, lvl=0):
    tabs = "\t" * lvl
    if d0.keys() != d1.keys():
        deld = set(d0.keys()) - set(d1.keys())
        addd = set(d1.keys()) - set(d0.keys())
        if deld:
            for k in sorted(deld):
                yield tabs + str(k) + " deleted"
        if addd:
            for k in sorted(addd):
                yield tabs + str(k) + " added"
    for k in sorted(set(d0.keys()).intersection(d1.keys())):
        if d0[k] != d1[k]:
            if isinstance(d0[k], dict) and isinstance(d1[k], dict):
                yield "{}{}:".format(tabs, k)
                yield from deepDictDiffIter(d0[k], d1[k], lvl+1)
            else:
                yield "{}{}: {} != {}".format(tabs, k, d0[k], d1[k])


class TestCase(unittest.TestCase):
    def assertDictEqual(self, d0, d1, msg=None):
        if d0 != d1:
            self.fail(self._formatMessage(
                msg,
                self._truncateMessage(
                    "Dicts not equal. Sizes {}, {}\n".format(
                        len(d0), len(d1)
                    ), "\n".join(deepDictDiffIter(d0, d1))
                )
            ))


class SimTest(TestCase):
    maxDiff = None
    def setUp(self):
        """Start an engine, install the sim module, and run it a while.

        This gives us some world-state to test upon.

        """
        from logging import getLogger, FileHandler
        self.engine = Engine(":memory:")
        logger = getLogger('LiSE.engine')
        logger.setLevel('DEBUG')
        logger.addHandler(FileHandler('test.log'))
        with self.engine.advancing:
            sim.install(self.engine)
        for i in range(72):
            self.engine.next_turn()
        self.engine.commit()

    def tearDown(self):
        """Close my engine."""
        self.engine.close()

    def testRulebooksCache(self):
        rulebooks = defaultdict(list)
        for (rulebook, rule) in self.engine.rule.query.rulebooks_rules():
            rulebooks[rulebook].append(rule)
        # Ignoring empty rulebooks because those only exist
        # implicitly, they don't have database records
        oldrulebooks = {}
        for (k, v) in self.engine._rulebooks_cache._data.items():
            if v:
                oldrulebooks[k] = [rule.name for rule in v]
        self.assertDictEqual(oldrulebooks, rulebooks)

    def testCharRulebooksCaches(self):
        charrb = {}
        for (
                character,
                character_rulebook,
                avatar_rulebook,
                character_thing_rulebook,
                character_place_rulebook,
                character_node_rulebook,
                character_portal_rulebook
        ) in self.engine.query.characters_rulebooks():
            charrb[character] = {
                'character': character_rulebook,
                'avatar': avatar_rulebook,
                'character_thing': character_thing_rulebook,
                'character_place': character_place_rulebook,
                'character_node': character_node_rulebook,
                'character_portal': character_portal_rulebook
            }
        self.assertDictEqual(
            charrb,
            self.engine._characters_rulebooks_cache._data
        )

    def testNodeRulebooksCache(self):
        noderb = defaultdict(dict)
        for (character, node, rulebook) in self.engine.query.nodes_rulebooks():
            noderb[character][node] = rulebook
        self.assertDictEqual(
            noderb,
            self.engine._nodes_rulebooks_cache._data
        )

    def testPortalRulebooksCache(self):
        portrb = StructuredDefaultDict(1, dict)
        for (character, nodeA, nodeB, rulebook) in self.engine.query.portals_rulebooks():
            portrb[character][nodeA][nodeB] = rulebook
        self.assertDictEqual(
            portrb,
            self.engine._portals_rulebooks_cache._data
        )

    def testAvatarnessCaches(self):
        user_avatarness = StructuredDefaultDict(3, WindowDict)
        for (character, graph, node, branch, tick, is_avatar) in self.engine.query.avatarness_dump():
            user_avatarness[graph][node][character][branch][tick] = is_avatar
        new_user_avatarness = StructuredDefaultDict(3, WindowDict)
        usr = self.engine._avatarness_cache.user_order
        for graph in usr:
            for node in usr[graph]:
                for char in usr[graph][node]:
                    if usr[graph][node][char]:
                        for branch in usr[graph][node][char]:
                            for tick, is_avatar in usr[graph][node][char][branch].items():
                                new_user_avatarness[graph][node][char][branch][tick] = is_avatar
        self.assertDictEqual(
            user_avatarness,
            new_user_avatarness
        )

    def testNodeRulesHandledCache(self):
        node_rules_handled_ticks = defaultdict(  # character:
            lambda: defaultdict(  # node:
                lambda: defaultdict(  # rulebook:
                    lambda: defaultdict(  # rule:
                        lambda: defaultdict(  # branch:
                            set  # ticks handled
                        )
                    )
                )
            )
        )
        new_node_rules_handled_ticks = defaultdict(  # character:
            lambda: defaultdict(  # node:
                lambda: defaultdict(  # rulebook:
                    lambda: defaultdict(  # rule:
                        dict
                    )
                )
            )
        )
        cache = self.engine._node_rules_handled_cache._data
        for char in cache:
            for node in cache[char]:
                for rulebook in cache[char][node]:
                    for rule in cache[char][node][rulebook]:
                        if cache[char][node][rulebook][rule]:
                            new_node_rules_handled_ticks[
                                char][node][rulebook][rule] \
                                = cache[char][node][rulebook][rule]
        for character, node, rulebook, rule, branch, tick in \
                self.engine.query.dump_node_rules_handled():
            node_rules_handled_ticks[
                character][node][rulebook][rule][branch].add(tick)
        self.assertDictEqual(
            node_rules_handled_ticks,
            new_node_rules_handled_ticks
        )

    def testPortalRulesHandledCache(self):
        portal_rules_handled_ticks = defaultdict(  # character:
            lambda: defaultdict(  # nodeA:
                lambda: defaultdict(  # nodeB:
                    lambda: defaultdict(  # rulebook:
                        lambda: defaultdict(  # rule:
                            lambda: defaultdict(  # branch:
                                set  # ticks handled
                            )
                        )
                    )
                )
            )
        )
        new_portal_rules_handled_ticks = defaultdict(  # character:
            lambda: defaultdict(  # nodeA:
                lambda: defaultdict(  # nodeB:
                    lambda: defaultdict(  # rulebook:
                        lambda: defaultdict(  # rule:
                            dict
                        )
                    )
                )
            )
        )
        cache = self.engine._portal_rules_handled_cache._data
        for character in cache:
            for nodeA in cache[character]:
                for nodeB in cache[character][nodeA]:
                    for rulebook in cache[character][nodeA][nodeB]:
                        for rule in cache[character][nodeA][nodeB][rulebook]:
                            if cache[character][nodeA][nodeB][rulebook][rule]:
                                new_portal_rules_handled_ticks[
                                    character][nodeA][nodeB][rulebook][rule] \
                                    = cache[character][nodeA][nodeB][
                                        rulebook][rule]
        for (character, nodeA, nodeB, idx, rulebook, rule, branch, tick) \
                in self.engine.query.dump_portal_rules_handled():
            portal_rules_handled_ticks[
                character][nodeA][nodeB][rulebook][rule][branch].add(tick)
        self.assertDictEqual(
            portal_rules_handled_ticks,
            new_portal_rules_handled_ticks
        )

    def testCharRulesHandledCaches(self):
        live = self.engine._character_rules_handled_cache._data
        for rulemap in [
                'character',
                'avatar',
                'character_thing',
                'character_place',
                'character_portal'
        ]:
            handled_ticks = StructuredDefaultDict(2, set)
            for character, rulebook, rule, branch, tick in getattr(
                    self.engine.query, 'handled_{}_rules'.format(rulemap)
            )():
                handled_ticks[character][rule][branch].add(tick)
            old_handled_ticks = StructuredDefaultDict(2, set)
            for character in live:
                if live[character][rulemap]:
                    for rule in live[character][rulemap]:
                        for branch, ticks in live[character][rulemap][rule].items():
                            self.assertIsInstance(ticks, set)
                            old_handled_ticks[character][rule][branch] = ticks
            self.assertDictEqual(
                old_handled_ticks,
                handled_ticks,
                "\n{} cache differs from DB".format(rulemap)
            )

    def testThingsCache(self):
        things = StructuredDefaultDict(3, tuple)
        for (character, thing, branch, tick, loc, nextloc) in \
                self.engine.query.things_dump():
            things[(character,)][thing][branch][tick] = (loc, nextloc)
        self.assertDictEqual(
            things,
            self.engine._things_cache.keys
        )

    def testRoommateCollisions(self):
        """Test queries' ability to tell that all of the students that share
        rooms have been in the same place.

        """
        done = set()
        for chara in self.engine.character.values():
            if chara.name in done:
                continue
            match = re.match('dorm(\d)room(\d)student(\d)', chara.name)
            if not match:
                continue
            dorm, room, student = match.groups()
            other_student = 1 if student == 0 else 0
            student = chara
            other_student = self.engine.character[
                'dorm{}room{}student{}'.format(dorm, room, other_student)
            ]
            
            same_loc_ticks = list(self.engine.ticks_when(
                student.avatar.only.historical('location')
                == other_student.avatar.only.historical('location')
            ))
            self.assertTrue(
                same_loc_ticks,
                "{} and {} don't seem to share a room".format(
                    student.name, other_student.name
                )
            )
            self.assertGreater(
                len(same_loc_ticks),
                6,
                "{} and {} share their room for less than 6 ticks".format(
                    student.name, other_student.name
                )
            )
            done.add(student.name)
            done.add(other_student.name)

    def testSoberCollisions(self):
        """Students that are neither lazy nor drunkards should all have been
        in class together at least once.

        """
        students = [
            stu for stu in
            self.engine.character['student_body'].stat['characters']
            if not (stu.stat['drunkard'] or stu.stat['lazy'])
        ]

        assert students

        def sameClasstime(stu0, stu1):
            self.assertTrue(
                self.engine.ticks_when(
                    stu0.avatar.only.historical('location') ==
                    stu1.avatar.only.historical('location') ==
                    self.engine.alias('classroom')
                ),
                "{stu0} seems not to have been in the classroom "
                "at the same time as {stu1}.\n"
                "{stu0} was there at ticks {ticks0}\n"
                "{stu1} was there at ticks {ticks1}".format(
                    stu0=stu0.name,
                    stu1=stu1.name,
                    ticks0=list(self.engine.ticks_when(stu0.avatar.only.historical('location') == self.engine.alias('classroom'))),
                    ticks1=list(self.engine.ticks_when(stu1.avatar.only.historical('location') == self.engine.alias('classroom')))
                )
            )
            return stu1

        reduce(sameClasstime, students)

    def testNoncollision(self):
        """Make sure students *not* from the same room never go there together"""
        dorm = defaultdict(lambda: defaultdict(dict))
        for character in self.engine.character.values():
            match = re.match('dorm(\d)room(\d)student(\d)', character.name)
            if not match:
                continue
            d, r, s = match.groups()
            dorm[d][r][s] = character
        for d in dorm:
            other_dorms = [dd for dd in dorm if dd != d]
            for r in dorm[d]:
                other_rooms = [rr for rr in dorm[d] if rr != r]
                for stu0 in dorm[d][r].values():
                    for rr in other_rooms:
                        for stu1 in dorm[d][rr].values():
                            self.assertFalse(
                                self.engine.ticks_when(
                                    stu0.avatar.only.historical('location') ==
                                    stu1.avatar.only.historical('location') ==
                                    self.engine.alias('dorm{}room{}'.format(d, r))
                                ),
                                "{} seems to share a room with {}".format(
                                    stu0.name, stu1.name
                                )
                            )
                    common = 'common{}'.format(d)
                    for dd in other_dorms:
                        for rr in dorm[dd]:
                            for stu1 in dorm[dd][rr].values():
                                self.assertFalse(
                                    self.engine.ticks_when(
                                        stu0.avatar.only.historical('location') ==
                                        stu1.avatar.only.historical('location') ==
                                        self.engine.alias(common)
                                    ),
                                    "{} seems to have been in the same"
                                    "common room  as {}".format(
                                        stu0.name, stu1.name
                                    )
                                )


def test_fast_delta():
    from LiSE.examples.kobold import inittest
    from LiSE.handle import EngineHandle
    hand = EngineHandle((':memory:',), {'random_seed': 69105})
    with hand._real.advancing:
        inittest(hand._real, shrubberies=20, kobold_sprint_chance=.9)
    # just set a baseline for the diff
    hand.get_slow_delta()
    ret, diff = hand.next_turn()
    slowd = hand.get_slow_delta()
    assert diff == slowd, "Fast delta differs from slow delta"
    ret, diff2 = hand.time_travel('trunk', 0, 0)
    slowd2 = hand.get_slow_delta()
    assert diff2 == slowd2, "Fast delta differs from slow delta"
    ret, diff3 = hand.time_travel('trunk', 3)
    slowd3 = hand.get_slow_delta()
    assert diff3 == slowd3, "Fast delta differs from slow delta"
    ret, diff4 = hand.time_travel('trunk', 1)
    slowd4 = hand.get_slow_delta()
    assert diff4 == slowd4, "Fast delta differs from slow delta"


def test_assignment():
    from LiSE.examples.college import install
    from LiSE.handle import EngineHandle
    hand = EngineHandle((':memory:',), {'random_seed': 69105})
    eng = hand._real
    with eng.advancing:
        install(eng)
    physical_inital_copy = {'edge_val': {
        'classroom': {'common0': {'is_mirror': True}, 'common1': {'is_mirror': True}, 'common2': {'is_mirror': True}},
        'dorm1room5': {'common1': {}}, 'dorm0room3': {'common0': {}}, 'dorm0room2': {'common0': {}},
        'dorm2room5': {'common2': {}}, 'dorm1room0': {'common1': {}},
        'common2': {'classroom': {}, 'dorm2room2': {'is_mirror': True}, 'dorm2room4': {'is_mirror': True},
                    'dorm2room0': {'is_mirror': True}, 'dorm2room5': {'is_mirror': True},
                    'dorm2room3': {'is_mirror': True}, 'dorm2room1': {'is_mirror': True}},
        'dorm0room4': {'common0': {}}, 'dorm2room0': {'common2': {}}, 'dorm0room1': {'common0': {}},
        'dorm0room5': {'common0': {}}, 'dorm1room3': {'common1': {}}, 'dorm2room4': {'common2': {}},
        'dorm2room1': {'common2': {}},
        'common1': {'classroom': {}, 'dorm1room2': {'is_mirror': True}, 'dorm1room0': {'is_mirror': True},
                    'dorm1room3': {'is_mirror': True}, 'dorm1room1': {'is_mirror': True},
                    'dorm1room5': {'is_mirror': True}, 'dorm1room4': {'is_mirror': True}},
        'dorm1room4': {'common1': {}}, 'dorm1room2': {'common1': {}}, 'dorm2room3': {'common2': {}},
        'common0': {'classroom': {}, 'dorm0room1': {'is_mirror': True}, 'dorm0room3': {'is_mirror': True},
                    'dorm0room2': {'is_mirror': True}, 'dorm0room5': {'is_mirror': True},
                    'dorm0room4': {'is_mirror': True}, 'dorm0room0': {'is_mirror': True}},
        'dorm1room1': {'common1': {}}, 'dorm2room2': {'common2': {}}, 'dorm0room0': {'common0': {}}}, 'hour': 0,
                            'node_val': {'classroom': {'rulebook': ('physical', 'classroom')},
                                         'dorm1room5': {'rulebook': ('physical', 'dorm1room5')},
                                         'dorm0room4student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room4student0'),
                                                                'location': 'dorm0room4'},
                                         'dorm0room3': {'rulebook': ('physical', 'dorm0room3')},
                                         'dorm0room2': {'rulebook': ('physical', 'dorm0room2')},
                                         'dorm2room5': {'rulebook': ('physical', 'dorm2room5')},
                                         'dorm1room2student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room2student1'),
                                                                'location': 'dorm1room2'},
                                         'dorm2room1student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room1student1'),
                                                                'location': 'dorm2room1'},
                                         'dorm0room0student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room0student0'),
                                                                'location': 'dorm0room0'},
                                         'dorm0room4': {'rulebook': ('physical', 'dorm0room4')},
                                         'dorm2room2student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room2student0'),
                                                                'location': 'dorm2room2'},
                                         'dorm0room1': {'rulebook': ('physical', 'dorm0room1')},
                                         'dorm1room3student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room3student0'),
                                                                'location': 'dorm1room3'},
                                         'dorm0room5': {'rulebook': ('physical', 'dorm0room5')},
                                         'dorm1room3student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room3student1'),
                                                                'location': 'dorm1room3'},
                                         'dorm1room3': {'rulebook': ('physical', 'dorm1room3')},
                                         'dorm2room2student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room2student1'),
                                                                'location': 'dorm2room2'},
                                         'dorm2room3student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room3student0'),
                                                                'location': 'dorm2room3'},
                                         'dorm1room4': {'rulebook': ('physical', 'dorm1room4')},
                                         'dorm2room0student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room0student1'),
                                                                'location': 'dorm2room0'},
                                         'dorm2room3student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room3student1'),
                                                                'location': 'dorm2room3'},
                                         'dorm2room4student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room4student1'),
                                                                'location': 'dorm2room4'},
                                         'dorm0room4student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room4student1'),
                                                                'location': 'dorm0room4'},
                                         'dorm1room4student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room4student1'),
                                                                'location': 'dorm1room4'},
                                         'dorm1room1': {'rulebook': ('physical', 'dorm1room1')},
                                         'dorm1room1student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room1student0'),
                                                                'location': 'dorm1room1'},
                                         'dorm0room5student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room5student0'),
                                                                'location': 'dorm0room5'},
                                         'dorm0room2student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room2student0'),
                                                                'location': 'dorm0room2'},
                                         'dorm1room0': {'rulebook': ('physical', 'dorm1room0')},
                                         'dorm1room0student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room0student1'),
                                                                'location': 'dorm1room0'},
                                         'dorm1room0student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room0student0'),
                                                                'location': 'dorm1room0'},
                                         'common2': {'rulebook': ('physical', 'common2')},
                                         'dorm2room2': {'rulebook': ('physical', 'dorm2room2')},
                                         'dorm2room5student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room5student1'),
                                                                'location': 'dorm2room5'},
                                         'dorm0room1student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room1student0'),
                                                                'location': 'dorm0room1'},
                                         'dorm2room0': {'rulebook': ('physical', 'dorm2room0')},
                                         'dorm2room4student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room4student0'),
                                                                'location': 'dorm2room4'},
                                         'dorm2room3': {'rulebook': ('physical', 'dorm2room3')},
                                         'dorm0room1student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room1student1'),
                                                                'location': 'dorm0room1'},
                                         'dorm1room5student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room5student0'),
                                                                'location': 'dorm1room5'},
                                         'dorm2room0student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room0student0'),
                                                                'location': 'dorm2room0'},
                                         'dorm2room1student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room1student0'),
                                                                'location': 'dorm2room1'},
                                         'dorm2room4': {'rulebook': ('physical', 'dorm2room4')},
                                         'dorm1room1student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room1student1'),
                                                                'location': 'dorm1room1'},
                                         'dorm0room0student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room0student1'),
                                                                'location': 'dorm0room0'},
                                         'dorm2room5student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm2room5student0'),
                                                                'location': 'dorm2room5'},
                                         'common1': {'rulebook': ('physical', 'common1')},
                                         'dorm1room4student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room4student0'),
                                                                'location': 'dorm1room4'},
                                         'dorm0room5student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room5student1'),
                                                                'location': 'dorm0room5'},
                                         'dorm0room3student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room3student1'),
                                                                'location': 'dorm0room3'},
                                         'dorm1room2': {'rulebook': ('physical', 'dorm1room2')},
                                         'dorm0room3student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room3student0'),
                                                                'location': 'dorm0room3'},
                                         'dorm0room0': {'rulebook': ('physical', 'dorm0room0')},
                                         'dorm1room5student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room5student1'),
                                                                'location': 'dorm1room5'},
                                         'dorm0room2student1': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm0room2student1'),
                                                                'location': 'dorm0room2'},
                                         'common0': {'rulebook': ('physical', 'common0')},
                                         'dorm2room1': {'rulebook': ('physical', 'dorm2room1')},
                                         'dorm1room2student0': {'next_location': None,
                                                                'rulebook': ('physical', 'dorm1room2student0'),
                                                                'location': 'dorm1room2'}},
                            'rulebooks': {'avatar': ('physical', 'avatar'), 'portal': ('physical', 'character_portal'),
                                          'place': ('physical', 'character_place'),
                                          'character': ('physical', 'character'),
                                          'node': ('physical', 'character_node'),
                                          'thing': ('physical', 'character_thing')}}
    physical_copy = hand.character_copy('physical')
    assert physical_copy == physical_inital_copy
    dorm_initial_copy = {'rulebooks': {'avatar': ('dorm0', 'avatar'), 'portal': ('dorm0', 'character_portal'),
                                       'place': ('dorm0', 'character_place'), 'character': ('dorm0', 'character'),
                                       'node': ('dorm0', 'character_node'), 'thing': ('dorm0', 'character_thing')},
                         'avatars': {'physical': frozenset(['dorm0room1', 'dorm0room3', 'dorm0room2', 'common0', 'dorm0room5',
                                                  'dorm0room4', 'dorm0room0'])}}
    dorm_copy = hand.character_copy('dorm0')
    assert dorm_copy == dorm_initial_copy
    physical = eng.character['physical']
    student_initial_copy = {'node_val': {
        'cell11': {'rulebook': ('dorm0room0student0', 'cell11'), 'drunk': 0, 'slow': 0},
        'cell43': {'rulebook': ('dorm0room0student0', 'cell43'), 'drunk': 0, 'slow': 0},
        'cell30': {'rulebook': ('dorm0room0student0', 'cell30'), 'drunk': 0, 'slow': 0},
        'cell38': {'rulebook': ('dorm0room0student0', 'cell38'), 'drunk': 0, 'slow': 0},
        'cell64': {'rulebook': ('dorm0room0student0', 'cell64'), 'drunk': 0, 'slow': 0},
        'cell12': {'rulebook': ('dorm0room0student0', 'cell12'), 'drunk': 0, 'slow': 0},
        'cell89': {'rulebook': ('dorm0room0student0', 'cell89'), 'drunk': 0, 'slow': 0},
        'cell96': {'rulebook': ('dorm0room0student0', 'cell96'), 'drunk': 0, 'slow': 0},
        'cell44': {'rulebook': ('dorm0room0student0', 'cell44'), 'drunk': 0, 'slow': 0},
        'cell22': {'rulebook': ('dorm0room0student0', 'cell22'), 'drunk': 0, 'slow': 0},
        'cell91': {'rulebook': ('dorm0room0student0', 'cell91'), 'drunk': 0, 'slow': 0},
        'cell85': {'rulebook': ('dorm0room0student0', 'cell85'), 'drunk': 0, 'slow': 0},
        'cell39': {'rulebook': ('dorm0room0student0', 'cell39'), 'drunk': 0, 'slow': 0},
        'cell29': {'rulebook': ('dorm0room0student0', 'cell29'), 'drunk': 0, 'slow': 0},
        'cell52': {'rulebook': ('dorm0room0student0', 'cell52'), 'drunk': 0, 'slow': 0},
        'cell15': {'rulebook': ('dorm0room0student0', 'cell15'), 'drunk': 0, 'slow': 0},
        'cell60': {'rulebook': ('dorm0room0student0', 'cell60'), 'drunk': 0, 'slow': 0},
        'cell50': {'rulebook': ('dorm0room0student0', 'cell50'), 'drunk': 0, 'slow': 0},
        'cell82': {'rulebook': ('dorm0room0student0', 'cell82'), 'drunk': 0, 'slow': 0},
        'cell24': {'rulebook': ('dorm0room0student0', 'cell24'), 'drunk': 0, 'slow': 0},
        'cell66': {'rulebook': ('dorm0room0student0', 'cell66'), 'drunk': 0, 'slow': 0},
        'cell88': {'rulebook': ('dorm0room0student0', 'cell88'), 'drunk': 0, 'slow': 0},
        'cell3': {'rulebook': ('dorm0room0student0', 'cell3'), 'drunk': 0, 'slow': 0},
        'cell83': {'rulebook': ('dorm0room0student0', 'cell83'), 'drunk': 0, 'slow': 0},
        'cell1': {'rulebook': ('dorm0room0student0', 'cell1'), 'drunk': 0, 'slow': 0},
        'cell73': {'rulebook': ('dorm0room0student0', 'cell73'), 'drunk': 0, 'slow': 0},
        'cell79': {'rulebook': ('dorm0room0student0', 'cell79'), 'drunk': 0, 'slow': 0},
        'cell0': {'rulebook': ('dorm0room0student0', 'cell0'), 'drunk': 0, 'slow': 0},
        'cell8': {'rulebook': ('dorm0room0student0', 'cell8'), 'drunk': 0, 'slow': 0},
        'cell26': {'rulebook': ('dorm0room0student0', 'cell26'), 'drunk': 0, 'slow': 0},
        'cell45': {'rulebook': ('dorm0room0student0', 'cell45'), 'drunk': 0, 'slow': 0},
        'cell34': {'rulebook': ('dorm0room0student0', 'cell34'), 'drunk': 0, 'slow': 0},
        'cell31': {'rulebook': ('dorm0room0student0', 'cell31'), 'drunk': 0, 'slow': 0},
        'cell55': {'rulebook': ('dorm0room0student0', 'cell55'), 'drunk': 0, 'slow': 0},
        'cell63': {'rulebook': ('dorm0room0student0', 'cell63'), 'drunk': 0, 'slow': 0},
        'cell80': {'rulebook': ('dorm0room0student0', 'cell80'), 'drunk': 0, 'slow': 0},
        'cell93': {'rulebook': ('dorm0room0student0', 'cell93'), 'drunk': 0, 'slow': 0},
        'cell81': {'rulebook': ('dorm0room0student0', 'cell81'), 'drunk': 0, 'slow': 0},
        'cell75': {'rulebook': ('dorm0room0student0', 'cell75'), 'drunk': 0, 'slow': 0},
        'cell46': {'rulebook': ('dorm0room0student0', 'cell46'), 'drunk': 0, 'slow': 0},
        'cell19': {'rulebook': ('dorm0room0student0', 'cell19'), 'drunk': 0, 'slow': 0},
        'cell6': {'rulebook': ('dorm0room0student0', 'cell6'), 'drunk': 0, 'slow': 0},
        'cell36': {'rulebook': ('dorm0room0student0', 'cell36'), 'drunk': 0, 'slow': 0},
        'cell32': {'rulebook': ('dorm0room0student0', 'cell32'), 'drunk': 0, 'slow': 0},
        'cell40': {'rulebook': ('dorm0room0student0', 'cell40'), 'drunk': 0, 'slow': 0},
        'cell47': {'rulebook': ('dorm0room0student0', 'cell47'), 'drunk': 0, 'slow': 0},
        'cell4': {'rulebook': ('dorm0room0student0', 'cell4'), 'drunk': 0, 'slow': 0},
        'cell28': {'rulebook': ('dorm0room0student0', 'cell28'), 'drunk': 0, 'slow': 0},
        'cell70': {'rulebook': ('dorm0room0student0', 'cell70'), 'drunk': 0, 'slow': 0},
        'cell14': {'rulebook': ('dorm0room0student0', 'cell14'), 'drunk': 0, 'slow': 0},
        'cell7': {'rulebook': ('dorm0room0student0', 'cell7'), 'drunk': 0, 'slow': 0},
        'cell53': {'rulebook': ('dorm0room0student0', 'cell53'), 'drunk': 0, 'slow': 0},
        'cell58': {'rulebook': ('dorm0room0student0', 'cell58'), 'drunk': 0, 'slow': 0},
        'cell69': {'rulebook': ('dorm0room0student0', 'cell69'), 'drunk': 0, 'slow': 0},
        'cell97': {'rulebook': ('dorm0room0student0', 'cell97'), 'drunk': 0, 'slow': 0},
        'cell57': {'rulebook': ('dorm0room0student0', 'cell57'), 'drunk': 0, 'slow': 0},
        'cell77': {'rulebook': ('dorm0room0student0', 'cell77'), 'drunk': 0, 'slow': 0},
        'cell48': {'rulebook': ('dorm0room0student0', 'cell48'), 'drunk': 0, 'slow': 0},
        'cell33': {'rulebook': ('dorm0room0student0', 'cell33'), 'drunk': 0, 'slow': 0},
        'cell74': {'rulebook': ('dorm0room0student0', 'cell74'), 'drunk': 0, 'slow': 0},
        'cell41': {'rulebook': ('dorm0room0student0', 'cell41'), 'drunk': 0, 'slow': 0},
        'cell65': {'rulebook': ('dorm0room0student0', 'cell65'), 'drunk': 0, 'slow': 0},
        'cell42': {'rulebook': ('dorm0room0student0', 'cell42'), 'drunk': 0, 'slow': 0},
        'cell94': {'rulebook': ('dorm0room0student0', 'cell94'), 'drunk': 0, 'slow': 0},
        'cell9': {'rulebook': ('dorm0room0student0', 'cell9'), 'drunk': 0, 'slow': 0},
        'cell13': {'rulebook': ('dorm0room0student0', 'cell13'), 'drunk': 0, 'slow': 0},
        'cell49': {'rulebook': ('dorm0room0student0', 'cell49'), 'drunk': 0, 'slow': 0},
        'cell99': {'rulebook': ('dorm0room0student0', 'cell99'), 'drunk': 0, 'slow': 0},
        'cell98': {'rulebook': ('dorm0room0student0', 'cell98'), 'drunk': 0, 'slow': 0},
        'cell23': {'rulebook': ('dorm0room0student0', 'cell23'), 'drunk': 0, 'slow': 0},
        'cell17': {'rulebook': ('dorm0room0student0', 'cell17'), 'drunk': 0, 'slow': 0},
        'cell72': {'rulebook': ('dorm0room0student0', 'cell72'), 'drunk': 0, 'slow': 0},
        'cell10': {'rulebook': ('dorm0room0student0', 'cell10'), 'drunk': 0, 'slow': 0},
        'cell56': {'rulebook': ('dorm0room0student0', 'cell56'), 'drunk': 0, 'slow': 0},
        'cell35': {'rulebook': ('dorm0room0student0', 'cell35'), 'drunk': 0, 'slow': 0},
        'cell78': {'rulebook': ('dorm0room0student0', 'cell78'), 'drunk': 0, 'slow': 0},
        'cell90': {'rulebook': ('dorm0room0student0', 'cell90'), 'drunk': 0, 'slow': 0},
        'cell18': {'rulebook': ('dorm0room0student0', 'cell18'), 'drunk': 0, 'slow': 0},
        'cell16': {'rulebook': ('dorm0room0student0', 'cell16'), 'drunk': 0, 'slow': 0},
        'cell61': {'rulebook': ('dorm0room0student0', 'cell61'), 'drunk': 0, 'slow': 0},
        'cell5': {'rulebook': ('dorm0room0student0', 'cell5'), 'drunk': 0, 'slow': 0},
        'cell51': {'rulebook': ('dorm0room0student0', 'cell51'), 'drunk': 0, 'slow': 0},
        'cell95': {'rulebook': ('dorm0room0student0', 'cell95'), 'drunk': 0, 'slow': 0},
        'cell87': {'rulebook': ('dorm0room0student0', 'cell87'), 'drunk': 0, 'slow': 0},
        'cell20': {'rulebook': ('dorm0room0student0', 'cell20'), 'drunk': 0, 'slow': 0},
        'cell37': {'rulebook': ('dorm0room0student0', 'cell37'), 'drunk': 0, 'slow': 0},
        'cell59': {'rulebook': ('dorm0room0student0', 'cell59'), 'drunk': 0, 'slow': 0},
        'cell92': {'rulebook': ('dorm0room0student0', 'cell92'), 'drunk': 0, 'slow': 0},
        'cell86': {'rulebook': ('dorm0room0student0', 'cell86'), 'drunk': 0, 'slow': 0},
        'cell54': {'rulebook': ('dorm0room0student0', 'cell54'), 'drunk': 0, 'slow': 0},
        'cell25': {'rulebook': ('dorm0room0student0', 'cell25'), 'drunk': 0, 'slow': 0},
        'cell76': {'rulebook': ('dorm0room0student0', 'cell76'), 'drunk': 0, 'slow': 0},
        'cell68': {'rulebook': ('dorm0room0student0', 'cell68'), 'drunk': 0, 'slow': 0},
        'cell84': {'rulebook': ('dorm0room0student0', 'cell84'), 'drunk': 0, 'slow': 0},
        'cell67': {'rulebook': ('dorm0room0student0', 'cell67'), 'drunk': 0, 'slow': 0},
        'cell71': {'rulebook': ('dorm0room0student0', 'cell71'), 'drunk': 0, 'slow': 0},
        'cell2': {'rulebook': ('dorm0room0student0', 'cell2'), 'drunk': 0, 'slow': 0},
        'cell21': {'rulebook': ('dorm0room0student0', 'cell21'), 'drunk': 0, 'slow': 0},
        'cell62': {'rulebook': ('dorm0room0student0', 'cell62'), 'drunk': 0, 'slow': 0},
        'cell27': {'rulebook': ('dorm0room0student0', 'cell27'), 'drunk': 0, 'slow': 0}}, 'lazy': True, 'roommate'
    : eng.character['dorm0room0student1'], 'xp': 0, 'rulebooks': {'avatar': ('dorm0room0student0', 'avatar'),
                                                                  'portal': ('dorm0room0student0', 'character_portal'),
                                                                  'place': ('dorm0room0student0', 'character_place'),
                                                                  'character': ('dorm0room0student0', 'character'),
                                                                  'node': ('dorm0room0student0', 'character_node'),
                                                                  'thing': ('dorm0room0student0', 'character_thing')},
        'room':
            physical.place['dorm0room0'], 'drunkard': False, 'avatars': {'physical': frozenset(['dorm0room0student0'])}}
    assert hand.character_copy('dorm0room0student0') == student_initial_copy


if __name__ == '__main__':
    unittest.main()