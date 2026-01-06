from pyodide_js import loadPackage
from pyscript import document, fetch

await loadPackage("micropip")
import micropip

await micropip.install("annotated-types")

from lisien import Engine
from lisien.examples import sickle

with Engine(workers=0) as eng:
	sickle.install(eng)
	for i in range(50):
		print(i)
		eng.next_turn()
	eng.export("sickle")

with Engine.from_archive("sickle.lisien", workers=0) as eng:
	print(list(eng.character))
	print("critters alive:")
	for node in eng.character["physical"].thing:
		print(node)
