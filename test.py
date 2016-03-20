import unittest
import LiSE
import re
from collections import defaultdict
from LiSE.engine import crhandled_defaultdict
from examples import college as sim


def deepDictDiffIter(d0, d1, lvl=0):
    tabs = "\t" * lvl
    if d0.keys() != d1.keys():
        deld = set(d0.keys()) - set(d1.keys())
        addd = set(d1.keys()) - set(d0.keys())
        if deld:
            for k in deld:
                yield tabs + str(k) + " deleted"
        if addd:
            for k in addd:
                yield tabs + str(k) + " added"
    for k in set(d0.keys()).intersection(d1.keys()):
        if d0[k] != d1[k]:
            if isinstance(d0[k], dict) and isinstance(d1[k], dict):
                yield tabs + k + ":"
                yield from deepDictDiffIter(d0[k], d1[k], lvl+1)
            else:
                yield tabs + k + ": {} != {}".format(d0[k], d1[k])


class LiSETest(unittest.TestCase):
    def setUp(self):
        """Start an engine, install the sim module, and run it a while.

        This gives us some world-state to test upon.

        """
        self.engine = LiSE.Engine(":memory:")
        sim.install(self.engine)
        for i in range(72):
            self.engine.next_tick()
        self.engine.commit()

    def assertDictEqual(self, d0, d1, msg=None):
        if d0 != d1:
            self.fail(self._formatMessage(
                msg,
                "Dicts not equal\n" + "\n".join(deepDictDiffIter(d0, d1))
            ))

    def tearDown(self):
        """Close my engine."""
        self.engine.close()

    def testCharRulebooksCaches(self):
        rulebooks = defaultdict(list)
        for (rulebook, rule) in self.engine.rule.db.rulebooks_rules():
            rulebooks[rulebook].append(rule)
        self.assertDictEqual(rulebooks, self.engine._rulebooks_cache)
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
            self.engine._nodes_rulebooks_cache[character][node],
            noderb
        )

    def testPortalRulebooksCache(self):
        portrb = defaultdict(
            lambda: defaultdict(dict)
        )
        for (character, nodeA, nodeB, rulebook) in self.engine.db.portals_rulebooks():
            portrb[character][nodeA][nodeB] = rulebook
        self.assertDictEqual(
            self.engine._portals_rulebooks_cache[character][nodeA][nodeB],
            portrb
        )

    def testAvatarnessCaches(self):
        db_avatarness = defaultdict(  # character:
            lambda: defaultdict(  # graph:
                lambda: defaultdict(  # node:
                    lambda: defaultdict(  # branch:
                        dict  # tick: is_avatar
                    )
                )
            )
        )
        user_avatarness = defaultdict(  # graph:
            lambda: defaultdict(  # node:
                lambda: defaultdict(  # character:
                    lambda: defaultdict(  # branch:
                        dict  # tick: is_avatar
                    )
                )
            )
        )
        for (character, graph, node, branch, tick, is_avatar) in self.engine.db.avatarness_dump():
            db_avatarness[character][graph][node][branch][tick] = is_avatar
            user_avatarness[graph][node][character][branch][tick] = is_avatar
        self.assertDictEqual(
            db_avatarness,
            self.engine._avatarness_cache.db_order
        )
        self.assertDictEqual(
            user_avatarness,
            self.engine._avatarness_cache.db_order
        )

    def testActiveRulesCache(self):
        actrules = defaultdict(  # rulebook:
            lambda: defaultdict(  # rule:
                lambda: defaultdict(  # branch:
                    dict  # tick: active
                )
            )
        )
        for rulebook, rule, branch, tick, active in self.engine.db.dump_active_rules():
            actrules[rulebook][rule][branch][tick] = active
        self.assertDictEqual(
            actrules,
            self.engine._active_rules_cache
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
        for character, node, rulebook, rule, branch, tick in \
                self.engine.db.dump_node_rules_handled():
            node_rules_handled_ticks[
                character][node][rulebook][rule][branch].add(tick)
        self.assertDictEqual(
            self.engine._node_rules_handled_cache,
            node_rules_handled_ticks
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
        for (character, nodeA, nodeB, idx, rulebook, rule, branch, tick) \
                in self.engine.db.dump_portal_rules_handled():
            portal_rules_handled_ticks[
                character][nodeA][nodeB][rulebook][rule][branch].add(tick)
        self.assertDictEqual(
            self.engine._portal_rules_handled_cache,
            portal_rules_handled_ticks
        )

    def testCharRulesHandledCaches(self):
        for rulemap in [
                'character',
                'avatar',
                'character_thing',
                'character_place',
                'character_portal'
        ]:
            handled_ticks = crhandled_defaultdict()
            for character, rulebook, rule, branch, tick in getattr(
                    self.engine.db, 'handled_{}_rules'.format(rulemap)
            ):
                handled_ticks[character][rulebook][rule][branch].add(tick)
            self.assertDictEqual(
                handled_ticks,
                getattr(self.engine, '_{}_rules_handled_cache'.format(rulemap))
            )

    def testThingsCache(self):
        things = defaultdict(  # character:
            lambda: defaultdict(  # thing:
                lambda: defaultdict(  # branch:
                    dict  # tick: (location, next_location)
                )
            )
        )
        for (character, thing, branch, tick, loc, nextloc) in \
                self.engine.db.things_dump():
            things[character][thing][branch][tick] = (loc, nextloc)
        self.assertDictEqual(
            things,
            self.engine._things_cache
        )

    def testRoommateCollisions(self):
        """Test queries' ability to tell that all of the students that share rooms have been in the same place."""
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
            other_student = self.engine.character['dorm{}room{}student{}'.format(dorm, room, other_student)]
            self.assertEqual(
                student.avatar.historical('location'),
                other_student.avatar.historical('location'),
                "Roommates don't seem to share a room: {}, {}".format(student, other_student)
            )
            done.add(student.name)
            done.add(other_student.name)


if __name__ == '__main__':
    unittest.main()
