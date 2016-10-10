from LiSE.examples.college import install
from LiSE.engine import Engine

def test():
    eng = Engine(":memory:")
    install(eng)
    for i in range(24):
        eng.next_tick()


if __name__ == '__main__':
    cProfile.run('test()')
