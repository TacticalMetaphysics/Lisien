from LiSE import Engine
from LiSE.examples import college, kobold, polygons, sickle


def test_college(engy):
    college.install(engy)
    engy.turn = 10  # wake up the students
    engy.next_turn()


def test_kobold(engy):
    kobold.inittest(engy, shrubberies=20, kobold_sprint_chance=.9)
    for i in range(10):
        engy.next_turn()


def test_polygons(engy):
    polygons.install(engy)
    for i in range(10):
        engy.next_turn()


def test_char_stat_startup(tempdir):
    with Engine(tempdir) as eng:
        tri = eng.new_character('triangle')
        sq = eng.new_character('square')

        sq.stat['min_sameness'] = 0.1
        assert 'min_sameness' in sq.stat
        sq.stat['max_sameness'] = 0.9
        assert 'max_sameness' in sq.stat
        tri.stat['min_sameness'] = 0.2
        assert 'min_sameness' in tri.stat
        tri.stat['max_sameness'] = 0.8
        assert 'max_sameness' in tri.stat

    with Engine(tempdir) as eng:
        assert 'min_sameness' in eng.character['square'].stat
        assert 'max_sameness' in eng.character['square'].stat
        assert 'min_sameness' in eng.character['triangle'].stat
        assert 'max_sameness' in eng.character['triangle'].stat


def test_sickle(engy):
    sickle.install(engy)
    for i in range(100):
        engy.next_turn()
