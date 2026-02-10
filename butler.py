import os
import sys
import tomllib

with open("lisien/pyproject.toml", "rb") as inf:
	cfg = tomllib.load(inf)

version = cfg["project"]["version"]

for lisien_wheel in os.listdir("lisien/dist"):
	if lisien_wheel.endswith(".whl"):
		break
else:
	sys.exit("Couldn't find the lisien wheel")
for elide_wheel in os.listdir("elide/dist"):
	if elide_wheel.endswith(".whl"):
		break
else:
	sys.exit("Couldn't find the elide wheel")
for elide_apk in os.listdir("bin/"):
	if elide_apk.startswith("Elide") and elide_apk.endswith(
		"-arm64-v8a_armeabi-v7a-debug.apk"
	):
		break
else:
	sys.exit("Couldn't find the Elide APK")

if not os.path.exists("lisien_windows"):
	sys.exit("Couldn't find the Elide Windows distribution")
if "ITCH_TOKEN" in os.environ:
	with open("butler_creds", "wt") as butler_creds:
		butler_creds.write(os.environ["ITCH_TOKEN"])
	postfix = " -i butler_creds"
else:
	postfix = ""
os.system(
	f"butler push lisien/dist/{lisien_wheel} clayote/lisien:lisien-whl --userversion {version}" + postfix
)
os.system(
	f"butler push elide/dist/{elide_wheel} clayote/lisien:elide-whl --userversion {version}" + postfix
)
os.system(
	f"butler push {windist} clayote/lisien:windows --userversion {version}" + postfix
)
os.system(
	f"butler push bin/{elide_apk} clayote/lisien:android --userversion {version}" + postfix
)
