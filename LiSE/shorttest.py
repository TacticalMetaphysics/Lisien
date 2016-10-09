from LiSE.examples.college import install
from LiSE.engine import Engine


eng = Engine(":memory:")
install(eng)
for i in range(24):
    eng.next_tick()
