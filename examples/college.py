# This file is part of LiSE, a framework for life simulation games.
# Copyright (c) Zachary Spector,  zacharyspector@gmail.com
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
cells made useless per drink, recovering as time passes).

This script will initialize LiSEworld.db and LiSEcode.db to run the
simulation described. To view it, run ELiDE from the same directory
as you ran this script from.

"""
import LiSE


eng = LiSE.Engine('LiSEworld.db', 'LiSEcode.db')
phys = eng.new_character('physical')
phys.stat['hour'] = 0


@phys.rule(always=True)  # runs every tick regardless of the situation
def time_passes(engine, character):
    character.stat['hour'] = (character.stat['hour'] + 1) % 24


# There's a character with all of the students in it, to make it easy to apply rules to all students.
student_body = eng.new_character('student_body')


classroom = phys.new_place('classroom')


@student_body.avatar.rule
def go_to_class(engine, character, node):
    # There's just one really long class every day.
    node.travel_to(engine.character['physical'].place['classroom'])


@go_to_class.trigger
def absent(engine, character, node):
    return node.location != engine.character['physical'].place['classroom']


@go_to_class.prereq
def class_in_session(engine, *args):
    return 8 <= engine.character['physical'].stat['hour'] < 15


@go_to_class.prereq
def be_timely(engine, character, node):
    # Even lazy students have a 50% chance of going to class every hour.
    #
    # We need to access the student character like this because the
    # ``character`` passed into the function is ``student_body``, where the
    # rule was assigned.
    #
    # Or we could have put the 'lazy' stat onto the node instead of the
    # character... or kept the student character in a stat of the node...
    # or assigned this rule to the student directly.
    for user in node.user.values():
        if user.name not in (character, 'student_body'):
            return not user.stat['lazy'] or engine.coinflip()


@student_body.avatar.rule
def leave_class(engine, character, node):
    for user in node.user.values():
        if user != character:
            node.travel_to(user.stat['room'])
            return


@leave_class.trigger
def in_classroom_after_class(engine, character, node):
    phys = engine.character['physical']
    return node.location == phys.place['classroom'] \
           and phys.stat['hour'] >= 15


# Let's make some rules and not assign them to anything yet.
@eng.rule
def drink(engine, character):
    braincells = list(character.node.values())
    engine.shuffle(braincells)
    for i in range(0, engine.randrange(1, 20)):
        braincells.pop()['drunk'] += 12


@drink.trigger
def party_time(engine, character):
    return 23 >= engine.character['physical'].stat['hour'] > 15


@drink.prereq
def is_drunkard(engine, character):
    return character.stat['drunkard']


@eng.rule
def sloth(engine, character):
    braincells = list(character.node.values())
    engine.shuffle(braincells)
    for i in range(0, engine.randrange(1, 20)):
        braincells.pop()['slow'] += 1


@sloth.trigger
def out_of_class(engine, character):
    # You don't want to use the global variable for the classroom
    # because it won't be around (or at least, won't work) after
    # the engine restarts.
    return character.avatar['physical'].location != \
        engine.character['physical'].place['classroom']

sloth.prereq(class_in_session)


@eng.rule
def learn(engine, character, node):
    character.stat['xp'] += 1


@learn.trigger
def in_class(engine, character, node):
    return character.avatar['physical'].location == \
        engine.character['physical'].place['classroom']

learn.prereq(class_in_session)


@learn.prereq
def pay_attention(engine, character, node):
    return node['drunk'] == node['slow'] == 0


@eng.rule
def sober_up(engine, character, node):
    node['drunk'] -= 1


@sober_up.trigger
def somewhat_drunk(engine, character, node):
    return node['drunk'] > 0


@eng.rule
def catch_up(engine, character, node):
    node['slow'] -= 1


@catch_up.trigger
def somewhat_late(engine, character, node):
    return node['slow'] > 0

catch_up.prereq(in_class)
catch_up.prereq(class_in_session)

# 3 dorms of 12 students each.
# Each dorm has 6 rooms.
# Modeling the teachers would be a logical way to extend this.
for n in range(0, 3):
    dorm = eng.new_character('dorm{}'.format(n))
    common = phys.new_place('common{}'.format(n))  # A common room for students to meet in
    dorm.add_avatar(common)
    common.two_way(classroom)
    # All rooms in a dorm are connected via its common room
    for i in range(0, 6):
        room = phys.new_place('dorm{}room{}'.format(n, i))
        dorm.add_avatar(room)
        room.two_way(common)
        student0 = eng.new_character('dorm{}room{}student0'.format(n, i))
        body0 = room.new_thing('dorm{}room{}student0'.format(n, i))
        student0.add_avatar(body0)
        assert (student0 in body0.user.values())
        student_body.add_avatar(body0)
        assert (student_body in body0.user.values())
        assert (student0 in body0.user.values())
        student1 = eng.new_character('dorm{}room{}student1'.format(n, i))
        body1 = room.new_thing('dorm{}room{}student1'.format(n, i))
        student1.add_avatar(body1)
        student_body.add_avatar(body1)
        student0.stat['room'] = student1.stat['room'] = room
        student0.stat['roommate'] = student1
        student1.stat['roommate'] = student0
        for student in (student0, student1):
            # Students' nodes are their brain cells.
            # They are useless if drunk or slow, but recover from both conditions a bit every hour.
            for k in range(0, 100):
                cell = student.new_node('cell{}'.format(k), drunk=0, slow=0)
                #  ``new_node`` is just an alias for ``new_place``;
                #  perhaps more logical when the places don't really
                #  represent potential locations
                student.stat['xp'] = 0
                student.stat['drunkard'] = eng.coinflip()
                student.stat['lazy'] = eng.coinflip()
            # Apply these previously written rules to each student
            for rule in (drink, sloth):
                student.rule(rule)
            # Apply these previously written rules to each brain cell
            for rule in (learn, sober_up, catch_up):
                student.node.rule(rule)

eng.close()
