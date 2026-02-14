"""Utility to remove repetitive, uninteresting lines from Kivy's Android logs"""

import re
from pathlib import Path

top = Path("kivylogs")
dull_regex = re.compile(
	r"\[DEBUG +\] +\nmeth:.*\n +sig.*\n +Public.*\n +Private.*\n +Protected.*\n +Static.*\n +Final.*\n +Synchronized.*\n +Volatile.*\n +Transient.*\n +Native.*\n +Interface.*\n +Abstract.*\n +Strict.*\n+"
)

for path in top.iterdir():
	if path.name.startswith("kivy"):
		with open(path, "rt") as inf:
			txt = inf.read()
		dull_strs = set(re.findall(dull_regex, txt))
		for dull_str in dull_strs:
			txt = txt.replace(dull_str, "")
		with open(path, "wt") as outf:
			outf.write(txt)
