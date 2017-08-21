


def time_slipping(engine, character, *, daylen: int, nightlen: int, twilightlen: float=0.0):
    if ('hour' not in character.stat):
        character.stat['hour'] = 0
        character.stat['day_period'] = ('dawn' if twilightlen else 'day')
        return
    twi_margin = (twilightlen / 2)
    hour = character.stat['hour'] = ((character.stat['hour'] + 1) % (daylen + nightlen))
    if twilightlen:
        if ((hour < twi_margin) or (hour > ((daylen + nightlen) - twi_margin))):
            character.stat['day_period'] = 'dawn'
        elif (twi_margin < hour < (daylen - twi_margin)):
            character.stat['day_period'] = 'day'
        elif ((daylen - twi_margin) < hour < (daylen + twi_margin)):
            character.stat['day_period'] = 'dusk'
        else:
            character.stat['day_period'] = 'night'
    else:
        character.stat['day_period'] = ('day' if (hour < daylen) else 'night')

def shrubsprint(engine, character, thing):
    print('shrub_places: {}'.format(thing['shrub_places']))
    shrub_places = sorted(list(thing['shrub_places']))
    if (thing['location'] in shrub_places):
        shrub_places.remove(thing['location'])
    print('shrub_places after: {}'.format(thing['shrub_places']))
    thing.travel_to(engine.choice(shrub_places))

def kill(engine, character, thing):
    character.thing['kobold'].delete()
    print('===KOBOLD DIES===')

def go2kobold(engine, character, thing):
    thing.travel_to(character.thing['kobold']['location'])

def wander(engine, character, thing):
    dests = sorted(list(character.place.keys()))
    dests.remove(thing['location'])
    thing.travel_to(engine.choice(dests))
