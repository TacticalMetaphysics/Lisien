import os
import re
import shutil
import tempfile
from collections import defaultdict

from lisien.engine import Engine

with tempfile.TemporaryDirectory() as tmp_path:
	shutil.unpack_archive(
		os.path.join(os.path.dirname(__file__), "college24_premade.tar.xz"),
		tmp_path,
	)
	with Engine(
		tmp_path,
		workers=0,
		connect_string=f"sqlite:///{tmp_path}/world.sqlite3",
	) as engine:
		dorm = defaultdict(lambda: defaultdict(dict))
		for character in engine.character.values():
			match = re.match(r"dorm(\d)room(\d)student(\d)", character.name)
			if not match:
				continue
			d, r, s = match.groups()
			dorm[d][r][s] = character
		for d in dorm:
			other_dorms = [dd for dd in dorm if dd != d]
			for r in dorm[d]:
				other_rooms = [rr for rr in dorm[d] if rr != r]
				for stu0 in dorm[d][r].values():
					for rr in other_rooms:
						for stu1 in dorm[d][rr].values():
							assert not list(
								engine.turns_when(
									stu0.unit.only.historical("location")
									== stu1.unit.only.historical("location")
									== "dorm{}room{}".format(d, r)
								)
							), "{} seems to share a room with {}".format(
								stu0.name, stu1.name
							)
							print(d, r, stu0, rr, stu1)
					common = "common{}".format(d)
					for dd in other_dorms:
						for rr in dorm[dd]:
							for stu1 in dorm[dd][rr].values():
								assert not list(
									engine.turns_when(
										stu0.unit.only.historical("location")
										== stu1.unit.only.historical(
											"location"
										)
										== common
									)
								), (
									"{} seems to have been in the same common room  as {}".format(
										stu0.name, stu1.name
									)
								)
								print(d, r, dd, rr, stu1)
