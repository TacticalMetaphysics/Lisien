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
