

def pure(fn):
    fn.pure = True
    return fn

@pure
def similar_neighbors(poly):
    'Trigger when my neighborhood fails to be enough like me'
    from operator import attrgetter
    home = poly.location
    similar = 0
    n = 0
    for (n, neighbor_home) in enumerate(map(attrgetter('destination'), home.portal.values()), 1):
        try:
            neighbor = next(iter(neighbor_home.contents()))
        except StopIteration:
            continue
        if (neighbor['shape'] == poly['shape']):
            similar += 1
    if (n == 0):
        return True
    return (poly.character.stat['min_sameness'] >= (similar / n))

@pure
def dissimilar_neighbors(poly):
    'Trigger when my neighborhood gets too much like me'
    from operator import attrgetter
    home = poly.location
    similar = 0
    n = 0
    for (n, neighbor_home) in enumerate(map(attrgetter('destination'), home.portal.values()), 1):
        try:
            neighbor = next(iter(neighbor_home.contents()))
        except StopIteration:
            continue
        if (neighbor['shape'] == poly['shape']):
            similar += 1
    if (n == 0):
        return True
    return (poly.character.stat['max_sameness'] < (similar / n))
