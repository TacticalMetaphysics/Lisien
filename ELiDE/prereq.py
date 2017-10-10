

def not_traveling(engine, character, thing):
    if (thing['next_location'] is not None):
        engine.info('kobold already travelling to {}'.format(thing['next_location']))
    return (thing['next_location'] is None)

def kobold_alive(engine, character, thing):
    return ('kobold' in character.thing)

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
