import unittest
import LiSE
import re
from examples import college as sim


class LiSETest(unittest.TestCase):
    def setUp(self):
        """Start an engine and install the sim module.

        This gives us some world-state to test upon.

        """
        self.engine = LiSE.Engine(":memory:")
        sim.install(self.engine)

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
            self.assertTrue(
                student.avatar.historical('location') == other_student.avatar.historical('location'),
                "Roommates don't seem to share a room: {}, {}".format(student, other_student)
            )
            done.add(student.name)
            done.add(other_student.name)


if __name__ == '__main__':
    unittest.main()
