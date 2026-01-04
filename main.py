from pyscript import document, fetch


CODEBERG_API_KEY = "a623f45a789dc58dc5f8bde889a9e1fefbba5f53"

print("fetching...")
fetching = await fetch(
	"https://codeberg.org/api/v1/packages/clayote?type=pypi",
	method="GET",
	headers={
		"Authorization": CODEBERG_API_KEY,
		"Accept": "application/json",
	},
)

try:
	if fetching.ok:
		interpreted = await fetching.text()
		print("fetched", str(interpreted))
	else:
		print("not ok", fetching.status)
except Exception as ex:
	print(ex)
	print(fetching)
	print("not fetched")
