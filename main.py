from pyscript import document
from pyscript.js_modules.compare_arrays import compare_lists


array0 = [1, 2, 4, 8, 15, 16, 23, 42]
array1 = [6, 5, 4, 3, 16, 16, 13, 42]

document.body.append(repr(list(compare_lists(array0, array1))))
