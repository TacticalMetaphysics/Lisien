"""A simulation of students and teachers, both living on campus,
attending classes and teaching or learning, as appropriate.

Learning is modeled in a way similar to the game Kudos 2: each
student has 100 "brain cells," and the teacher sends an experience
point to each, once per lesson. If the brain cell is awake and
alert, the student receives the experience point and puts it toward
leveling-up. Otherwise it's wasted.

Some students are slow and some are drunkards. These students will
randomly show up late (brain cells become useless in proportion to
how late, recovering immediately afterward) or drunk (some brain
cells made useless per drink, recovering as time passes). Other
students who happen to hang around with them have a chance of
adopting their behavior for the following day. 5 days slow or
drunk, and you become one of them. 5 days quick or sober, and
you lose the trait.

This script will initialize LiSEworld.db and LiSEcode.db to run the
simulation described. To view it, run ELiDE from the same directory
as you ran this script from.

"""
import LiSE


eng = LiSE.Engine('LiSEworld.db', 'LiSEcode.db')
phys = eng.new_character('physical')
phys.stat['hour'] = 0


@phys.rule(always=True)
def time_passes(engine, character):
    character.stat['hour'] = (character.stat['hour'] + 1) % 24

# There's a character with all of the students in it, to make it easy to apply rules to all students.
student_body = phys.new_character('student_body')


@student_body.avatar.rule
def sober_up(engine, character, node):
    node.stat['drunk'] -= 1


@sober_up.trigger
def drunken(engine, character, node):
    return node.stat['drunk'] >= 1


@student_body.avatar.rule
def catch_up(engine, character, node):
    node.stat['slow'] -= 1


@catch_up.trigger
def late(engine, character, node):
    return node.stat['slow'] >= 1

classroom = phys.new_node('classroom')


@student_body.avatar.rule
def go_to_class(engine, character, node):
    # There's just one really long class every day.
    node.travel_to(classroom)


@go_to_class.trigger
def class_time(engine, character, node):
    return phys.stat['hour'] == 8


@go_to_class.trigger
def not_in_class(engine, character, node):
    return (
        9 <= phys.stat['hour'] < 15 and
        node.location != classroom
    )


@go_to_class.prereq
def be_timely(engine, character, node):
    return not node['lazy'] or engine.coinflip()


@student_body.avatar.rule
def sloth(engine, character, node):
    node['slow'] += 1


sloth.trigger(not_in_class)

# 3 dorms of 12 students each.
# Each dorm has 6 rooms.
# Modeling the teachers would be a logical way to extend this.
for n in range(0, 3):
    dorm = eng.new_character('dorm{}'.format(n))
    common = phys.new_node('common{}'.format(n))  # A common room for students to meet in
    dorm.add_avatar(common)
    common.two_way(classroom)
    # All rooms in a dorm are connected via its common room
    for i in range(0, 6):
        room = phys.new_node('dorm{}room{}'.format(n, i))
        dorm.add_avatar(room)
        room.two_way(common)
        student0 = eng.new_character('dorm{}room{}student0'.format(n, i))
        body0 = room.new_thing('dorm{}room{}student0'.format(n, i))
        student0.add_avatar(body0)
        student_body.add_avatar(body0)
        student1 = eng.new_character('dorm{}room{}student1'.format(n, i))
        body1 = room.new_thing('dorm{}room{}student1')
        student1.add_avatar(body1)
        student_body.add_avatar(body1)
        student0.stat['room'] = student1.stat['room'] = room
        student0.stat['roommate'] = student1
        student1.stat['roommate'] = student0
        for student in (student0, student1):
            # Students' nodes are their brain cells.
            # Brain cells have stats that go down by one every hour, if above zero.
            for k in range(0, 100):
                cell = student.new_node('cell{}'.format(k), drunk=0, slow=0)
                student.stat['xp'] = 0
                student.stat['late'] = False
                student.stat['drunkard'] = eng.coinflip()
                student.stat['lazy'] = eng.coinflip()
            @student.node.rule
            def learn(engine, character, node):
                student.stat['xp'] += 1

            @learn.trigger
            def be_in_class(engine, character, node):
                return (
                    character.avatar.physical.location == classroom and
                    9 <= phys.stat['hour'] < 15
                )

            @learn.prereq
            def pay_attention(engine, character, node):
                return node['drunk'] == node['slow'] == 0