from tempfile import TemporaryDirectory

import kobold
import polygons

import wolfsheep
from lisien import Engine

RANDOM_SEED = 69105

with (
	TemporaryDirectory() as td,
	Engine(td, workers=0, random_seed=RANDOM_SEED) as eng,
):
	kobold.inittest(eng)
	eng.export("kobold")
with (
	TemporaryDirectory() as td,
	Engine(td, workers=0, random_seed=RANDOM_SEED) as eng,
):
	polygons.install(eng)
	eng.export("polygons")
with (
	TemporaryDirectory() as td,
	Engine(td, workers=0, random_seed=RANDOM_SEED) as eng,
):
	wolfsheep.install(eng, seed=RANDOM_SEED)
	eng.export("wolfsheep")
