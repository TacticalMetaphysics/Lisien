import re


def retick(s):
	"""Replace single backtick quotations with double backtick"""
	return re.sub(r"`(.+?)`", r"``\1``", s)


def main():
	with open("CHANGES.txt", "rt") as inf, open("CHANGES.rst", "wt") as outf:
		accumulator = []
		for line in inf:
			line = line.strip()
			if line.startswith("=="):
				if accumulator:
					outf.write(
						"- " + " ".join(map(retick, accumulator)) + "\n" * 3
					)
					accumulator.clear()
				date, version = line.strip("= ").split("|")
				outf.write(f"{version.strip()} ({date.strip()})\n")
				outf.write(
					"-" * (len(date.strip()) + len(version.strip()) + 3)
				)
				outf.write("\n\n")
			elif line.startswith("* "):
				if accumulator:
					outf.write(
						"- " + " ".join(map(retick, accumulator)) + "\n" * 2
					)
					accumulator.clear()
				accumulator.append(line.removeprefix("* "))
			elif line:
				accumulator.append(line)
		if accumulator:
			outf.write("- " + " ".join(map(retick, accumulator)))


if __name__ == "__main__":
	main()
