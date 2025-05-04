import os
import shutil

shutil.rmtree(
	".buildozer/android/platform/build-arm64-v8a_armeabi-v7a/dists/Elide",
	ignore_errors=True,
)
shutil.rmtree(".buildozer/android/app/elide", ignore_errors=True)
pardir = ".buildozer/android/platform/build-arm64-v8a_armeabi-v7a/build/python-installs/Elide/armeabi-v7a/"
for package in os.listdir(pardir):
	if package.startswith("lisien") or package.startswith("elide"):
		abspath = os.path.join(pardir, package)
		if (
			not os.path.exists(abspath)
			or not os.path.isdir(abspath)
			or os.path.islink(abspath)
		):
			continue
		shutil.rmtree(abspath)
