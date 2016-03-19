import unittest
import LiSE
import re
from examples import college as sim


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

    def tearDown(self):
        """Close my engine."""
        self.engine.close()

    def test_roommate_collisions(self):
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

    def test_rule_names_activeness(self):
        # TODO: test this for all RuleMapping subclasses
        phys = self.engine.character['physical']
        self.assertEqual(
            set(phys._rule_names_activeness()),
            set(
                getattr(self.engine.db, 'current_rules_' + phys.book)(
                    phys.name, *self.engine.time
                )
            )
        )

    def test_rulebook_name(self):
        # TODO: test this for all RuleMapping subclasses
        phys = self.engine.character['physical']
        self.assertEqual(
            self.engine.db.get_rulebook_char(
                phys._book,
                phys.name
            ), self.engine._characters_rulebooks_cache[
                phys.name][phys._book]
        )

    def test_active_rule_char(self):
        phys = self.engine.character['physical']
        for rule in phys.rule:
            self.assertTrue(self.engine.db.active_rule_char(
                phys._table,
                phys.name,
                phys.rulebook.name,
                rule,
                *self.engine.time
            ))

    def test_nodes_existence(self):
        phys = self.engine.character['physical']
        for node in phys.node:
            self.assertTrue(self.engine.db.node_exists(
                phys.name,
                node,
                *self.engine.time
            ))

    def test_avatarness(self):
        for char in self.engine.character.values():
            self.assertEqual(
                self.engine._avatarness_cache.db_order[char.name],
                self.engine.db.avatarness(
                    char.name,
                    *self.engine.time
                )
            )
            for graph in char.avatar:
                for av in char.avatar[graph]:
                    self.assertTrue(self.engine.db.is_avatar_of(
                        char.name,
                        graph,
                        av,
                        *self.engine.time
                    ))
            for (g, n, a) in self.engine.db.avatars_now(
                    char.name,
                    *self.engine.time
            ):
                if a:
                    self.assertIn(g, char.avatar)
                    self.assertIn(n, char.avatar[g])
                else:
                    if g in char.avatar:
                        self.assertNotIn(n, char.avatar[g])

    def test_avatarness_branchdata(self):
        for character in self.engine.character.values():
            if len(character.avatar) == 1:
                valueiter = iter([character.avatar])
            else:
                valueiter = iter(character.avatar.values())
            for avmap in valueiter:
                self.assertEqual(
                    list(avmap._branchdata(*self.engine.time)),
                    list(self.engine.db.avatar_branch_data(
                        character.name,
                        avmap.graph,
                        *self.engine.time
                    ))
                )

    def test_thingness(self):
        phys = self.engine.character['physical']
        for thing in phys.thing:
            self.assertTrue(self.engine.db.node_is_thing(
                phys.name,
                thing,
                *self.engine.time
            ))
        for place in phys.place:
            self.assertFalse(self.engine.db.node_is_thing(
                phys.name,
                place,
                *self.engine.time
            ))

    def test_rule_mapping(self):
        # TODO: every type of rule mapping
        for char in self.engine.character.values():
            self.assertEqual(
                list(char.rule),
                list(self.engine.db.rulebook_rules(char.rule.rulebook.name))
            )
            self.assertEqual(
                list(char.avatar.rule),
                list(self.engine.db.rulebook_rules(char.avatar.rule.rulebook.name))
            )
            self.assertEqual(
                list(char.thing.rule),
                list(self.engine.db.rulebook_rules(char.thing.rule.rulebook.name))
            )
            self.assertEqual(
                list(char.place.rule),
                list(self.engine.db.rulebook_rules(char.place.rule.rulebook.name))
            )
            self.assertEqual(
                list(char.node.rule),
                list(self.engine.db.rulebook_rules(char.node.rule.rulebook.name))
            )
            self.assertEqual(
                list(char.portal.rule),
                list(self.engine.db.rulebook_rules(char.portal.rule.rulebook.name))
            )
            for node in char.node:
                self.assertEqual(
                    list(node.rule),
                    list(self.engine.db.node_rules(
                        char.name,
                        node.name,
                        *self.engine.time
                    ))
                )
            for portal in char.portals():
                self.assertEqual(
                    list(portal.rule),
                    list(self.engine.db.portal_rules(
                        char.name,
                        portal['origin'],
                        portal['destination'],
                        *self.engine.time
                    ))
                )
            for av in char.avatar:
                chara = self.engine.character[av]
                for node in chara.node:
                    self.assertEqual(
                        list(node.rule),
                        list(self.engine.db.node_rules(
                            chara.name,
                            node.name,
                            *self.engine.time
                        ))
                    )
                for port in chara.portals():
                    self.assertEqual(
                        list(port.rule),
                        list(self.engine.db.portal_rules(
                            chara.name,
                            port['origin'],
                            port['destination'],
                            *self.engine.time
                        ))
                    )


if __name__ == '__main__':
    unittest.main()
