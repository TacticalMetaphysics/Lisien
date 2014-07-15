from LiSE import LiSE

lise = LiSE(
    world_filename='lise.world',
    code_filename='lise.code'
)

lise.tick = -1

phys = lise.new_character("Physical")
phys.add_place("Home")
phys.add_place("Work")
phys.add_portal("Home", "Work")
phys.add_portal("Work", "Home")
npc = lise.new_character("NonPlayer")
npc.add_avatar("npc", "Physical", "Home")

@npc.rule
def home2work(engine, npc, rule):
    """Rule to schedule a new trip to work by 9 o'clock."""
    # arrange to get to work by 9 o'clock
    print("Will go to work in 9 hours")
    npc.avatar["Physical"].travel_to_by(
        "Work",
        engine.tick+9)

h2w = npc.rule["home2work"]

@h2w.prereq
def daystart(engine, npc, rule):
    """Run at midnight only."""
    return engine.tick % 24 == 0

@h2w.prereq
def home_all_day(engine, npc, rule):
    """Run if I'm scheduled to be at Home for this tick and the
    following twenty-four.

    """
    present = engine.tick
    for t in xrange(present, present+24):
        # The branch and tick pointers will be reset by the
        # event handler once this function returns, don't
        # worry.
        engine.tick = t
        # I really only have one avatar, but doing it this way
        # I don't need to care what its name is, and it
        # handles the case where I have more than one avatar.
        for avatar in npc.avatar["Physical"].values():
            if avatar["location"] != "Home":
                return False
    print("Home all day.")
    return True

@npc.rule
def work2home(engine, npc, rule):
    """Rule to go home when work's over, at 5 o'clock."""
    # Leave, go home, arrive whenever
    print("Leaving work now.")
    npc.avatar["Physical"].travel_to("Home")

w2h = npc.rule["work2home"]

@w2h.prereq
def closing_time(engine, npc, rule):
    """Run at 5pm only."""
    return engine.tick % 24 == 17

@w2h.prereq
def at_work(engine, npc, rule):
    """Run only when I'm at Work."""
    return npc.avatar["Physical"]["location"] == "Work"

while lise.tick < 76:
    now = lise.tick
    lise.advance()
    if lise.tick != now:
        print("Location at tick {}: {}".format(lise.tick, npc.avatar["Physical"]["location"]))

lise.close()

lise = LiSE(
    world_filename='lise.world',
    code_filename='lise.code'
)
npc = lise.get_character("NonPlayer")

while lise.tick < 128:
    now = lise.tick
    lise.advance()
    if lise.tick != now:
        print("Location at tick {}: {}".format(lise.tick, npc.avatar["Physical"]["location"]))
