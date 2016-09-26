import unittest
from unittest.mock import MagicMock
import re
from functools import reduce
from collections import defaultdict
from networkx import DiGraph
from gorm.pickydict import StructuredDefaultDict
from gorm.window import WindowDict
from .engine import Engine
from .examples import college as sim


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
    def setUp(self):
        """Start an engine, install the sim module, and run it a while.

        This gives us some world-state to test upon.

        """
        self.engine = Engine(":memory:")
        sim.install(self.engine)
        for i in range(72):
            self.engine.next_tick()
        self.engine.commit()

    def tearDown(self):
        """Close my engine."""
        self.engine.close()

    def testRulebooksCache(self):
        rulebooks = defaultdict(list)
        for (rulebook, rule) in self.engine.rule.db.rulebooks_rules():
            rulebooks[rulebook].append(rule)
        # Ignoring empty rulebooks because those only exist
        # implicitly, they don't have database records
        oldrulebooks = {}
        for (k, v) in self.engine._rulebooks_cache.items():
            if v:
                oldrulebooks[k] = v
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
        ) in self.engine.db.characters_rulebooks():
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
            self.engine._characters_rulebooks_cache
        )

    def testNodeRulebooksCache(self):
        noderb = defaultdict(dict)
        for (character, node, rulebook) in self.engine.db.nodes_rulebooks():
            noderb[character][node] = rulebook
        self.assertDictEqual(
            noderb,
            self.engine._nodes_rulebooks_cache
        )

    def testPortalRulebooksCache(self):
        portrb = StructuredDefaultDict(1, dict)
        for (character, nodeA, nodeB, rulebook) in self.engine.db.portals_rulebooks():
            portrb[character][nodeA][nodeB] = rulebook
        self.assertDictEqual(
            portrb,
            self.engine._portals_rulebooks_cache
        )

    def testAvatarnessCaches(self):
        db_avatarness = StructuredDefaultDict(3, WindowDict)
        user_avatarness = StructuredDefaultDict(3, WindowDict)
        for (character, graph, node, branch, tick, is_avatar) in self.engine.db.avatarness_dump():
            db_avatarness[character][graph][node][branch][tick] = is_avatar
            user_avatarness[graph][node][character][branch][tick] = is_avatar
        new_db_avatarness = StructuredDefaultDict(3, WindowDict)
        db = self.engine._avatarness_cache.db_order
        for char in db:
            for graph in db[char]:
                for node in db[char][graph]:
                    if db[char][graph][node]:
                        for branch in db[char][graph][node]:
                            for tick, is_avatar in db[char][graph][node][branch].items():
                                new_db_avatarness[char][graph][node][branch][tick] = is_avatar
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
            db_avatarness,
            new_db_avatarness
        )
        self.assertDictEqual(
            user_avatarness,
            new_user_avatarness
        )

    def testActiveRulesCache(self):
        actrules = defaultdict(  # rulebook:
            lambda: defaultdict(  # rule:
                lambda: defaultdict(  # branch:
                    dict  # tick: active
                )
            )
        )
        newactrules = defaultdict(dict)
        cache = self.engine._active_rules_cache
        for rulebook in cache:
            for rule in cache[rulebook]:
                if cache[rulebook][rule]:
                    newactrules[rulebook][rule] = cache[rulebook][rule]
        for (
                rulebook, rule, branch, tick, active
        ) in self.engine.db.dump_active_rules():
            actrules[rulebook][rule][branch][tick] = active
        self.assertDictEqual(
            actrules,
            newactrules
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
        cache = self.engine._node_rules_handled_cache
        for char in cache:
            for node in cache[char]:
                for rulebook in cache[char][node]:
                    for rule in cache[char][node][rulebook]:
                        if cache[char][node][rulebook][rule]:
                            new_node_rules_handled_ticks[
                                char][node][rulebook][rule] \
                                = cache[char][node][rulebook][rule]
        for character, node, rulebook, rule, branch, tick in \
                self.engine.db.dump_node_rules_handled():
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
        cache = self.engine._portal_rules_handled_cache
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
                in self.engine.db.dump_portal_rules_handled():
            portal_rules_handled_ticks[
                character][nodeA][nodeB][rulebook][rule][branch].add(tick)
        self.assertDictEqual(
            portal_rules_handled_ticks,
            new_portal_rules_handled_ticks
        )

    def testCharRulesHandledCaches(self):
        for rulemap in [
                'character',
                'avatar',
                'character_thing',
                'character_place',
                'character_portal'
        ]:
            handled_ticks = StructuredDefaultDict(3, set)
            for character, rulebook, rule, branch, tick in getattr(
                    self.engine.db, 'handled_{}_rules'.format(rulemap)
            )():
                handled_ticks[character][rulebook][rule][branch].add(tick)
            old_handled_ticks = StructuredDefaultDict(3, set)
            live = getattr(
                self.engine, '_{}_rules_handled_cache'.format(rulemap)
            )
            for character in live:
                for rulebook in live[character]:
                    if live[character][rulebook]:
                        for rule in live[character][rulebook]:
                            for branch, ticks in live[character][rulebook][rule].items():
                                old_handled_ticks[character][rulebook][rule][branch] = ticks
            self.assertDictEqual(
                old_handled_ticks,
                handled_ticks,
                "\n{} cache differs from DB".format(rulemap)
            )

    def testThingsCache(self):
        things = StructuredDefaultDict(3, tuple)
        for (character, thing, branch, tick, loc, nextloc) in \
                self.engine.db.things_dump():
            things[character][thing][branch][tick] = (loc, nextloc)
        self.assertDictEqual(
            things,
            self.engine._things_cache
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
                student.avatar.historical('location')
                == other_student.avatar.historical('location')
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
                    stu0.avatar.historical('location') ==
                    stu1.avatar.historical('location') ==
                    self.engine.alias('classroom')
                ),
                "{stu0} seems not to have been in the classroom "
                "at the same time as {stu1}.\n"
                "{stu0} was there at ticks {ticks0}\n"
                "{stu1} was there at ticks {ticks1}".format(
                    stu0=stu0.name,
                    stu1=stu1.name,
                    ticks0=list(self.engine.ticks_when(stu0.avatar.historical('location') == self.engine.alias('classroom'))),
                    ticks1=list(self.engine.ticks_when(stu1.avatar.historical('location') == self.engine.alias('classroom')))
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
                                    stu0.avatar.historical('location') ==
                                    stu1.avatar.historical('location') ==
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
                                        stu0.avatar.historical('location') ==
                                        stu1.avatar.historical('location') ==
                                        self.engine.alias(common)
                                    ),
                                    "{} seems to have been in the same"
                                    "common room  as {}".format(
                                        stu0.name, stu1.name
                                    )
                                )


class BindingTestCase(TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'worlddb' not in kwargs:
            kwargs['worlddb'] = ':memory:'
        if 'codedb' not in kwargs:
            kwargs['codedb'] = ':memory:'
        self.kwargs = kwargs

    def setUp(self):
        self.engine = Engine(**self.kwargs)

    def tearDown(self):
        self.engine.close()

    def test_bind_string(self):
        """Test binding to the string store, and to a particular string"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()

        # these would normally be called using decorators but I don't
        # think I can do mocks that way
        self.engine.string.listener(general)
        self.engine.string.listener(specific, 'spam')
        self.engine.string.listener(string='ham')(inert)

        self.engine.string['spam'] = 'eggs'
        general.assert_called_once_with(self.engine.string, 'spam', 'eggs')
        specific.assert_called_once_with(self.engine.string, 'spam', 'eggs')
        general = MagicMock()
        specific = MagicMock()
        self.engine.string.listener(general)
        self.engine.string.listener(string='spam')(specific)
        del self.engine.string['spam']
        general.assert_called_once_with(self.engine.string, 'spam', None)
        specific.assert_called_once_with(self.engine.string, 'spam', None)
        self.assertEqual(inert.call_count, 0)
        bound = MagicMock()
        self.engine.string.lang_listener(bound)
        self.engine.string.language = 'jpn'
        bound.assert_called_once_with(self.engine.string, 'jpn')

    def test_bind_func_store(self):
        """Test binding to the function store, and to a specific function
        name

        """
        for store in self.engine.stores:
            general = MagicMock()
            specific = MagicMock()
            inert = MagicMock()
            getattr(self.engine, store).listener(general)
            getattr(self.engine, store).listener(name='spam')(specific)
            getattr(self.engine, store).listener(name='ham')(inert)

            def nothing():
                pass
            getattr(self.engine, store)['spam'] = nothing
            general.assert_called_once_with(
                getattr(self.engine, store),
                'spam',
                nothing
            )
            specific.assert_called_once_with(
                getattr(self.engine, store),
                'spam',
                nothing
            )
            del getattr(self.engine, store)['spam']
            general.assert_called_with(
                getattr(self.engine, store),
                'spam',
                None
            )
            specific.assert_called_with(
                getattr(self.engine, store),
                'spam',
                None
            )
            self.assertEqual(general.call_count, 2)
            self.assertEqual(specific.call_count, 2)
            self.assertEqual(inert.call_count, 0)

    def test_bind_univ_var(self):
        """Test binding to the universal variable store, and to a specific
        var

        """
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        self.engine.universal.listener(general)
        self.engine.universal.listener(key='spam')(specific)
        self.engine.universal.listener(key='ham')(inert)
        self.engine.universal['spam'] = 'eggs'
        general.assert_called_once_with(
            self.engine.universal,
            'spam',
            'eggs'
        )
        specific.assert_called_once_with(
            self.engine.universal,
            'spam',
            'eggs'
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_map(self):
        """Test binding to the CharacterMapping, and to a specific character"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        self.engine.character.listener(general)
        self.engine.character.listener(specific, 'spam')
        self.engine.character.listener(inert, 'ham')
        self.engine.character['spam'] = DiGraph(eggs=True)
        general.assert_called_once_with(
            self.engine.character,
            'spam',
            self.engine.character['spam']
        )
        specific.assert_called_once_with(
            self.engine.character,
            'spam',
            self.engine.character['spam']
        )
        self.assertEqual(inert.call_count, 0)
        self.assertTrue(self.engine.character['spam'].stat['eggs'])

    def test_bind_char_thing(self):
        """Test binding to a character's thing mapping, and to a specific
        thing

        """
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        self.engine.character['spam'] = DiGraph(eggs=True)
        char = self.engine.character['spam']
        self.assertTrue(char.stat['eggs'])
        # I have to put the thing someplace
        char.place['plate'] = {'flat': True}
        char.thing.listener(general)
        char.thing.listener(specific, 'baked_beans')
        char.thing['baked_beans'] = {'location': 'plate'}
        th = char.thing['baked_beans']
        self.assertEqual(th['location'], 'plate')
        general.assert_called_once_with(
            char.thing,
            'baked_beans',
            th
        )
        specific.assert_called_once_with(
            char.thing,
            'baked_beans',
            th
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_place(self):
        """Test binding to the place mapping of a character"""
        self.engine.character['spam'] = DiGraph()
        ch = self.engine.character['spam']
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        ch.place.listener(general)
        ch.place.listener(place='plate')(specific)
        ch.place.listener(inert, 'floor')
        ch.place['plate'] = {'flat': True}
        pl = ch.place['plate']
        self.assertTrue(pl['flat'])
        general.assert_called_once_with(
            ch.place,
            'plate',
            pl
        )
        specific.assert_called_once_with(
            ch.place,
            'plate',
            pl
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_portal(self):
        """Test binding to character's portal mapping"""
        self.engine.character['spam'] = DiGraph()
        ch = self.engine.character['spam']
        ch.place['kitchen'] = {'smell': 'yummy'}
        ch.place['porch'] = {'rustic': True}
        nodeA = ch.place['kitchen']
        nodeB = ch.place['porch']
        generalA = MagicMock()
        specificA = MagicMock()
        inert = MagicMock()
        ch.portal.listener(generalA)
        ch.portal.listener(specificA, 'kitchen')
        ch.portal.listener(inert, 'living_room')
        generalB = MagicMock()
        specificB = MagicMock()
        ch.portal['kitchen'].listener(generalB)
        ch.portal['kitchen'].listener(specificB, 'porch')
        ch.portal['kitchen'].listener(inert, 'balcony')
        ch.portal['kitchen']['porch'] = {'locked': False}
        port = ch.portal['kitchen']['porch']
        generalB.assert_called_once_with(
            ch.portal['kitchen'],
            nodeA,
            nodeB,
            port
        )
        specificB.assert_called_once_with(
            ch.portal['kitchen'],
            nodeA,
            nodeB,
            port
        )
        self.assertFalse(port['locked'])
        generalA.assert_called_once_with(
            ch.portal,
            nodeA,
            nodeB,
            port
        )
        specificA.assert_called_once_with(
            ch.portal,
            nodeA,
            nodeB,
            port
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_avatar(self):
        """Test binding to ``add_avatar``"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        if 'a' not in self.engine.character:
            self.engine.add_character('a')
        if 'b' not in self.engine.character:
            self.engine.add_character('b')
        chara = self.engine.character['a']
        charb = self.engine.character['b']
        chara.avatar_listener(general)
        chara.avatar_listener(specific, 'b')
        chara.avatar_listener(inert, 'z')
        pl = charb.new_place('q')
        chara.add_avatar(pl)
        general.assert_called_once_with(
            chara,
            charb,
            pl,
            True
        )
        specific.assert_called_once_with(
            chara,
            charb,
            pl,
            True
        )
        chara.del_avatar(pl)
        general.assert_called_with(
            chara,
            charb,
            pl,
            False
        )
        specific.assert_called_with(
            chara,
            charb,
            pl,
            False
        )
        charc = self.engine.new_character('c')
        plc = charc.new_place('c')
        chara.add_avatar(plc)
        general.assert_called_with(
            chara,
            charc,
            plc,
            True
        )
        self.assertEqual(general.call_count, 3)
        self.assertEqual(specific.call_count, 2)
        self.assertEqual(inert.call_count, 0)

    def test_bind_rule(self):
        """Test binding to a rulemap and a rule therein"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        if 'a' not in self.engine.character:
            self.engine.add_character('a')
        char = self.engine.character['a']
        char.rule.listener(general)
        char.rule.listener(specific, 'spam')
        char.rule.listener(inert, 'eggs')

        @char.rule
        def spam(*args):
            pass

        general.assert_called_once_with(
            char.rule,
            spam,
            True
        )
        specific.assert_called_once_with(
            char.rule,
            spam,
            True
        )
        del char.rule['spam']
        general.assert_called_with(
            char.rule,
            spam,
            False
        )
        specific.assert_called_with(
            char.rule,
            spam,
            False
        )
        self.engine.rule.listener(general)
        self.engine.rule.listener(specific, 'ham')
        self.engine.rule.listener(inert, 'eggs')
        @self.engine.rule
        def ham(*args):
            pass
        general.assert_called_with(
            self.engine.rule,
            ham,
            True
        )
        specific.assert_called_with(
            self.engine.rule,
            ham,
            True
        )
        del self.engine.rule['ham']
        general.assert_called_with(
            self.engine.rule,
            ham,
            False
        )
        specific.assert_called_with(
            self.engine.rule,
            ham,
            False
        )
        @self.engine.rule
        def baked_beans(*args):
            pass
        general.assert_called_with(
            self.engine.rule,
            baked_beans,
            True
        )
        self.assertEqual(general.call_count, 5)
        self.assertEqual(specific.call_count, 4)
        self.assertEqual(inert.call_count, 0)

    def test_bind_rule_funlist(self):
        """Test binding to each of the function lists of a rule"""
        trig = MagicMock()
        preq = MagicMock()
        act = MagicMock()
        if 'a' not in self.engine.character:
            self.engine.add_character('a')
        ch = self.engine.character['a']

        @ch.rule
        def nothing(*args):
            pass

        nothing.triggers.listener(trig)
        nothing.prereqs.listener(preq)
        nothing.actions.listener(act)

        def something(*args):
            pass

        nothing.trigger(something)
        nothing.prereq(something)
        nothing.action(something)
        trig.assert_called_once_with(nothing.triggers)
        preq.assert_called_once_with(nothing.prereqs)
        act.assert_called_once_with(nothing.actions)

    def test_bind_char_stat(self):
        """Test binding to a character's stat and to all of its stats"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        char = self.engine.new_character('c')
        char.stat.listener(general)
        char.stat.listener(stat='spam')(specific)
        char.stat.listener(inert, 'ham')
        char.stat['spam'] = 'tasty'
        general.assert_called_with(
            char,
            'spam',
            'tasty'
        )
        specific.assert_called_with(
            char,
            'spam',
            'tasty'
        )
        char.stat['eggs'] = 'eggy'
        general.assert_called_with(
            char,
            'eggs',
            'eggy'
        )
        del char.stat['spam']
        general.assert_called_with(
            char,
            'spam',
            None
        )
        specific.assert_called_with(
            char,
            'spam',
            None
        )
        del char.stat['eggs']
        general.assert_called_with(
            char,
            'eggs',
            None
        )
        self.assertEqual(general.call_count, 4)
        self.assertEqual(specific.call_count, 2)
        self.assertEqual(inert.call_count, 0)

    def test_bind_place_stat(self):
        """Test binding to one of a place's stats, and to all of them"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        char = self.engine.new_character('c')
        pl = char.new_place('p')
        pl.listener(general)
        pl.listener(stat='spam')(specific)
        pl.listener(inert, 'eggs')
        pl['spam'] = 'tasty'
        general.assert_called_once_with(
            pl,
            'spam',
            'tasty'
        )
        specific.assert_called_once_with(
            pl,
            'spam',
            'tasty'
        )
        pl['baked_beans'] = 'tastier'
        general.assert_called_with(
            pl,
            'baked_beans',
            'tastier'
        )
        del pl['spam']
        general.assert_called_with(
            pl,
            'spam',
            None
        )
        specific.assert_called_with(
            pl,
            'spam',
            None
        )
        self.assertEqual(general.call_count, 3)
        self.assertEqual(specific.call_count, 2)
        self.assertEqual(inert.call_count, 0)

    def test_bind_thing_stat(self):
        """Test binding to a thing's stat"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        char = self.engine.new_character('c')
        char.add_place('pl')
        th = char.new_thing('th', location='pl')
        th.listener(general)
        th.listener(specific, 'spam')
        th.listener(inert, 'ham')
        th['spam'] = 'tasty'
        general.assert_called_once_with(
            th,
            'spam',
            'tasty'
        )
        specific.assert_called_once_with(
            th,
            'spam',
            'tasty'
        )
        th['eggs'] = 'less_tasty'
        general.assert_called_with(
            th,
            'eggs',
            'less_tasty'
        )
        del th['spam']
        general.assert_called_with(
            th,
            'spam',
            None
        )
        specific.assert_called_with(
            th,
            'spam',
            None
        )
        self.assertEqual(general.call_count, 3)
        self.assertEqual(specific.call_count, 2)
        self.assertEqual(inert.call_count, 0)

if __name__ == '__main__':
    unittest.main()
