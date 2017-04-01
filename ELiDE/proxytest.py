from LiSE.proxy import EngineProcessManager

procman = EngineProcessManager()
eng = procman.start(':memory:')


phys = eng.new_character('physical')
place = phys.new_place('place1')
place['heresy'] = True
assert place['heresy']
eng.tick = 1
assert place['heresy']
place['heresy'] = False
assert not place['heresy']
eng.tick = 0
assert place['heresy']
eng.tick = 1
assert not place['heresy']
