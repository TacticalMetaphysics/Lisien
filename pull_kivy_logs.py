import os
import subprocess


filenames = list(
	filter(
		None,
		subprocess.run(
			[
				"adb",
				"shell",
				"run-as org.tacmeta.elide ls files/app/.kivy/logs",
			],
			stdout=subprocess.PIPE,
		)
		.stdout.decode()
		.split("\n"),
	)
)
os.makedirs("kivylogs", exist_ok=True)
for fn in filenames:
	with open(os.path.join("kivylogs", fn[:-3] + "log"), "wb") as outf:
		outf.write(
			subprocess.run(
				[
					"adb",
					"shell",
					"run-as org.tacmeta.elide cat files/app/.kivy/logs/" + fn,
				],
				stdout=subprocess.PIPE,
			).stdout
		)

lisien_log = subprocess.run(
	[
		"adb",
		"shell",
		"run-as org.tacmeta.elide sh -c 'if [ -r files/app/lisien.log ]; then echo 1; fi;'",
	],
	stdout=subprocess.PIPE,
).stdout
if lisien_log:
	with open(os.path.join("kivylogs", "lisien.log"), "wb") as outf:
		outf.write(lisien_log)
