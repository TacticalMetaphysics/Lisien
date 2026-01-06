from pyodide_js import loadPackage
from pyscript import document, fetch

await loadPackage("micropip")
import micropip

await micropip.install("annotated-types")

from lisien import types


CODEBERG_API_KEY = "a623f45a789dc58dc5f8bde889a9e1fefbba5f53"
print(dir(types))
