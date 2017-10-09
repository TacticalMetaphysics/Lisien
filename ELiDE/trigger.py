

def truth(engine, character):
    return True

def uncovered(engine, character, thing):
    for shrub_candidate in thing.location.contents():
        if (shrub_candidate.name[:5] == 'shrub'):
            return False
    engine.info('kobold uncovered')
    return True

def breakcover(engine, character, thing):
    if (engine.random() < thing['sprint_chance']):
        engine.info('kobold breaking cover')
        return True

def sametile(engine, character, thing):
    try:
        return (thing['location'] == character.thing['kobold']['location'])
    except KeyError:
        return False

def aware(engine, character, thing):
    from math import hypot
    try:
        bold = character.thing['kobold']
    except KeyError:
        return False
    (dx, dy) = bold['location']
    (ox, oy) = thing['location']
    xdist = abs((dx - ox))
    ydist = abs((dy - oy))
    dist = hypot(xdist, ydist)
    return (dist <= thing['sight_radius'])

def standing_still(engine, character, thing):
    return (thing['next_location'] is None)
