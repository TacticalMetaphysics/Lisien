import os
import shutil

shutil.rmtree(
	".buildozer/android/platform/build-arm64-v8a_armeabi-v7a/dists/Elide",
	ignore_errors=True,
)
shutil.rmtree(".buildozer/android/app/elide", ignore_errors=True)
pardirs = []
prefix = ".buildozer/android/platform/"
for plat in os.listdir(prefix):
	if "build" not in os.listdir(os.path.join(prefix, plat)):
		continue
	for build_dir in os.listdir(os.path.join(prefix, plat, "build")):
		if build_dir not in {"other_builds", "python-installs"}:
			continue
		if build_dir == "other_builds":
			pardirs.append(os.path.join(prefix, plat, "build", build_dir))
		elif build_dir == "python-installs":
			pre = os.path.join(prefix, plat, "build", build_dir, "Elide")
			if os.path.exists(pre) and os.path.isdir(pre):
				pardirs.extend(os.path.join(pre, arch) for arch in os.listdir(pre))
print("looking for built packages in", pardirs)
for pardir in pardirs:
	for package in os.listdir(pardir):
		if package.lower().startswith("lisien") or package.lower().startswith("elide"):
			abspath = os.path.join(pardir, package)
			if (
				not os.path.exists(abspath)
				or not os.path.isdir(abspath)
				or os.path.islink(abspath)
			):
				continue
			print("removing", abspath)
			shutil.rmtree(abspath)
