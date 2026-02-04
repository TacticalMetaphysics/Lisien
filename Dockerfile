FROM buildpack-deps:trixie

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# cannot remove LANG even though https://bugs.python.org/issue19846 is fixed
# last attempted removal of LANG broke many users:
# https://github.com/docker-library/python/pull/570
ENV LANG C.UTF-8

# runtime dependencies
RUN set -eux; \
    mkdir -pm755 /etc/apt/keyrings; \
    wget -O - https://dl.winehq.org/wine-builds/winehq.key | gpg --dearmor -o /etc/apt/keyrings/winehq-archive.key; \
    wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/debian/dists/trixie/winehq-trixie.sources; \
	dpkg --add-architecture i386; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		libbluetooth-dev \
		tk-dev \
		uuid-dev \
		xvfb \
		cmake \
		ninja-build \
		libx11-dev \
		libwayland-dev \
		libgles-dev \
		winehq-stable \
		openjdk-21-jdk-headless \
		autoconf \
		automake \
		build-essential \
		ccache \
		cmake \
		gettext \
		git \
		libffi-dev \
		libltdl-dev \
		libssl-dev \
		libtool \
		patch \
		pkg-config \
		unzip \
		zip \
		zlib1g-dev \
		libalsaplayer-dev \
		libasound2-dev \
		libfluidsynth-dev \
	; \
	apt-get dist-clean; \
	COMMIT_HASH=b0084151dee976c74891c459fd6fc0f27bb249d4; \
	wget -O kivy.zip https://github.com/kivy/kivy/archive/$COMMIT_HASH.zip; \
	unzip kivy.zip; \
	cd kivy-$COMMIT_HASH; \
    bash tools/build_linux_dependencies.sh;

ENV GPG_KEY 7169605F62C751356D054A26A821E680E5FA6305
ENV PYTHON_VERSION 3.12.12
ENV PYTHON_SHA256 fb85a13414b028c49ba18bbd523c2d055a30b56b18b92ce454ea2c51edc656c4

RUN set -eux; \
	\
	wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
	echo "$PYTHON_SHA256 *python.tar.xz" | sha256sum -c -; \
	wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"; \
	GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
	gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY"; \
	gpg --batch --verify python.tar.xz.asc python.tar.xz; \
	gpgconf --kill all; \
	rm -rf "$GNUPGHOME" python.tar.xz.asc; \
	mkdir -p /usr/src/python; \
	tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
	rm python.tar.xz; \
	\
	cd /usr/src/python; \
	gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
	./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-optimizations \
		--enable-option-checking=fatal \
		--enable-shared \
		$(test "${gnuArch%%-*}" != 'riscv64' && echo '--with-lto') \
		--with-ensurepip \
	; \
	nproc="$(nproc)"; \
	EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
	LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
	arch="$(dpkg --print-architecture)"; arch="${arch##*-}"; \
# https://docs.python.org/3.12/howto/perf_profiling.html
# https://github.com/docker-library/python/pull/1000#issuecomment-2597021615
	case "$arch" in \
		amd64|arm64) \
			# only add "-mno-omit-leaf" on arches that support it
			# https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/x86-Options.html#index-momit-leaf-frame-pointer-2
			# https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/AArch64-Options.html#index-momit-leaf-frame-pointer
			EXTRA_CFLAGS="${EXTRA_CFLAGS:-} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"; \
			;; \
		i386) \
			# don't enable frame-pointers on 32bit x86 due to performance drop.
			;; \
		*) \
			# other arches don't support "-mno-omit-leaf"
			EXTRA_CFLAGS="${EXTRA_CFLAGS:-} -fno-omit-frame-pointer"; \
			;; \
	esac; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-}" \
	; \
# https://github.com/docker-library/python/issues/784
# prevent accidental usage of a system installed libpython of the same version
	rm python; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-} -Wl,-rpath='\$\$ORIGIN/../lib'" \
		python \
	; \
	make install; \
	\
# enable GDB to load debugging data: https://github.com/docker-library/python/pull/701
	bin="$(readlink -ve /usr/local/bin/python3)"; \
	dir="$(dirname "$bin")"; \
	mkdir -p "/usr/share/gdb/auto-load/$dir"; \
	cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
	\
	cd /; \
	rm -rf /usr/src/python; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
		\) -exec rm -rf '{}' + \
	; \
	\
	ldconfig; \
	\
	export PYTHONDONTWRITEBYTECODE=1; \
	python3 --version; \
	pip3 --version

ENV PYTHON_VERSION 3.13.12
ENV PYTHON_SHA256 2a84cd31dd8d8ea8aaff75de66fc1b4b0127dd5799aa50a64ae9a313885b4593

RUN set -eux; \
	\
	wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
	echo "$PYTHON_SHA256 *python.tar.xz" | sha256sum -c -; \
	wget -O python.tar.xz.asc "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz.asc"; \
	GNUPGHOME="$(mktemp -d)"; export GNUPGHOME; \
	gpg --batch --keyserver hkps://keys.openpgp.org --recv-keys "$GPG_KEY"; \
	gpg --batch --verify python.tar.xz.asc python.tar.xz; \
	gpgconf --kill all; \
	rm -rf "$GNUPGHOME" python.tar.xz.asc; \
	mkdir -p /usr/src/python; \
	tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
	rm python.tar.xz; \
	\
	cd /usr/src/python; \
	gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
	./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-optimizations \
		--enable-option-checking=fatal \
		--enable-shared \
		$(test "${gnuArch%%-*}" != 'riscv64' && echo '--with-lto') \
		--with-ensurepip \
	; \
	nproc="$(nproc)"; \
	EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
	LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
	arch="$(dpkg --print-architecture)"; arch="${arch##*-}"; \
# https://docs.python.org/3.12/howto/perf_profiling.html
# https://github.com/docker-library/python/pull/1000#issuecomment-2597021615
	case "$arch" in \
		amd64|arm64) \
			# only add "-mno-omit-leaf" on arches that support it
			# https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/x86-Options.html#index-momit-leaf-frame-pointer-2
			# https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/AArch64-Options.html#index-momit-leaf-frame-pointer
			EXTRA_CFLAGS="${EXTRA_CFLAGS:-} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"; \
			;; \
		i386) \
			# don't enable frame-pointers on 32bit x86 due to performance drop.
			;; \
		*) \
			# other arches don't support "-mno-omit-leaf"
			EXTRA_CFLAGS="${EXTRA_CFLAGS:-} -fno-omit-frame-pointer"; \
			;; \
	esac; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-}" \
	; \
# https://github.com/docker-library/python/issues/784
# prevent accidental usage of a system installed libpython of the same version
	rm python; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-} -Wl,-rpath='\$\$ORIGIN/../lib'" \
		python \
	; \
	make install; \
	\
# enable GDB to load debugging data: https://github.com/docker-library/python/pull/701
	bin="$(readlink -ve /usr/local/bin/python3)"; \
	dir="$(dirname "$bin")"; \
	mkdir -p "/usr/share/gdb/auto-load/$dir"; \
	cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
	\
	cd /; \
	rm -rf /usr/src/python; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
		\) -exec rm -rf '{}' + \
	; \
	\
	ldconfig; \
	\
	export PYTHONDONTWRITEBYTECODE=1; \
	python3 --version; \
	pip3 --version

ENV PYTHON_VERSION 3.14.3
ENV PYTHON_SHA256 a97d5549e9ad81fe17159ed02c68774ad5d266c72f8d9a0b5a9c371fe85d902b

RUN set -eux; \
	\
	savedAptMark="$(apt-mark showmanual)"; \
	apt-get update; \
	apt-get install -y --no-install-recommends \
		libzstd-dev \
	; \
	\
	wget -O python.tar.xz "https://www.python.org/ftp/python/${PYTHON_VERSION%%[a-z]*}/Python-$PYTHON_VERSION.tar.xz"; \
	echo "$PYTHON_SHA256 *python.tar.xz" | sha256sum -c -; \
	mkdir -p /usr/src/python; \
	tar --extract --directory /usr/src/python --strip-components=1 --file python.tar.xz; \
	rm python.tar.xz; \
	\
	cd /usr/src/python; \
	gnuArch="$(dpkg-architecture --query DEB_BUILD_GNU_TYPE)"; \
	./configure \
		--build="$gnuArch" \
		--enable-loadable-sqlite-extensions \
		--enable-optimizations \
		--enable-option-checking=fatal \
		--enable-shared \
		$(test "${gnuArch%%-*}" != 'riscv64' && echo '--with-lto') \
		--with-ensurepip \
	; \
	nproc="$(nproc)"; \
	EXTRA_CFLAGS="$(dpkg-buildflags --get CFLAGS)"; \
	LDFLAGS="$(dpkg-buildflags --get LDFLAGS)"; \
	arch="$(dpkg --print-architecture)"; arch="${arch##*-}"; \
# https://docs.python.org/3.12/howto/perf_profiling.html
# https://github.com/docker-library/python/pull/1000#issuecomment-2597021615
	case "$arch" in \
		amd64|arm64) \
			# only add "-mno-omit-leaf" on arches that support it
			# https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/x86-Options.html#index-momit-leaf-frame-pointer-2
			# https://gcc.gnu.org/onlinedocs/gcc-14.2.0/gcc/AArch64-Options.html#index-momit-leaf-frame-pointer
			EXTRA_CFLAGS="${EXTRA_CFLAGS:-} -fno-omit-frame-pointer -mno-omit-leaf-frame-pointer"; \
			;; \
		i386) \
			# don't enable frame-pointers on 32bit x86 due to performance drop.
			;; \
		*) \
			# other arches don't support "-mno-omit-leaf"
			EXTRA_CFLAGS="${EXTRA_CFLAGS:-} -fno-omit-frame-pointer"; \
			;; \
	esac; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-}" \
	; \
# https://github.com/docker-library/python/issues/784
# prevent accidental usage of a system installed libpython of the same version
	rm python; \
	make -j "$nproc" \
		"EXTRA_CFLAGS=${EXTRA_CFLAGS:-}" \
		"LDFLAGS=${LDFLAGS:-} -Wl,-rpath='\$\$ORIGIN/../lib'" \
		python \
	; \
	make install; \
	\
# enable GDB to load debugging data: https://github.com/docker-library/python/pull/701
	bin="$(readlink -ve /usr/local/bin/python3)"; \
	dir="$(dirname "$bin")"; \
	mkdir -p "/usr/share/gdb/auto-load/$dir"; \
	cp -vL Tools/gdb/libpython.py "/usr/share/gdb/auto-load/$bin-gdb.py"; \
	\
	cd /; \
	rm -rf /usr/src/python; \
	\
	find /usr/local -depth \
		\( \
			\( -type d -a \( -name test -o -name tests -o -name idle_test \) \) \
			-o \( -type f -a \( -name '*.pyc' -o -name '*.pyo' -o -name 'libpython*.a' \) \) \
		\) -exec rm -rf '{}' + \
	; \
	\
	ldconfig; \
	\
	apt-mark auto '.*' > /dev/null; \
	apt-mark manual $savedAptMark; \
	find /usr/local -type f -executable -not \( -name '*tkinter*' \) -exec ldd '{}' ';' \
		| awk '/=>/ { so = $(NF-1); if (index(so, "/usr/local/") == 1) { next }; gsub("^/(usr/)?", "", so); printf "*%s\n", so }' \
		| sort -u \
		| xargs -rt dpkg-query --search \
# https://manpages.debian.org/bookworm/dpkg/dpkg-query.1.en.html#S (we ignore diversions and it'll be really unusual for more than one package to provide any given .so file)
		| awk 'sub(":$", "", $1) { print $1 }' \
		| sort -u \
		| xargs -r apt-mark manual \
	; \
	apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=false; \
	apt-get dist-clean; \
	\
	export PYTHONDONTWRITEBYTECODE=1; \
	python3 --version; \
	pip3 --version

# download latest butler from itch
RUN <<EOF
	set -eux
	wget -O butler.zip https://broth.itch.zone/butler/linux-amd64/LATEST/archive.zip
	unzip butler.zip
	mv butler /usr/local/bin/
	mv *.so /usr/local/lib/
EOF
# patch kivy
RUN patch -p1 -d kivy-* <<EOF
diff --git a/kivy/uix/recycleboxlayout.py b/kivy/uix/recycleboxlayout.py
index 12d7d387f..3e0cbb7b6 100644
--- a/kivy/uix/recycleboxlayout.py
+++ b/kivy/uix/recycleboxlayout.py
@@ -108,7 +108,7 @@ class RecycleBoxLayout(RecycleLayout, BoxLayout):
             self.minimum_size = l + r, t + b
             return

-        view_opts = self.view_opts
+        view_opts = self.converted_opts()
         n = len(view_opts)
         for i, x, y, w, h in self._iterate_layout(
                 [(opt['size'], opt['size_hint'], opt['pos_hint'],
diff --git a/kivy/uix/recyclelayout.py b/kivy/uix/recyclelayout.py
index 90c3a5011..115f586e9 100644
--- a/kivy/uix/recyclelayout.py
+++ b/kivy/uix/recyclelayout.py
@@ -266,6 +266,13 @@ class RecycleLayout(RecycleLayoutManagerBehavior, Layout):
                 self.clear_layout()

         assert len(data) == len(opts)
+
+        for i, item in enumerate(data):
+            if len(opts) > i and opts[i] is not None:
+                continue
+            opts[i] = self._convert_opt(item)
+
+    def _convert_opt(self, item):
         ph_key = self.key_pos_hint
         ph_def = self.default_pos_hint
         sh_key = self.key_size_hint
@@ -279,50 +286,47 @@ class RecycleLayout(RecycleLayoutManagerBehavior, Layout):
         viewcls_def = self.viewclass
         viewcls_key = self.key_viewclass
         iw, ih = self.initial_size
-
-        sh = []
-        for i, item in enumerate(data):
-            if opts[i] is not None:
-                continue
-
-            ph = ph_def if ph_key is None else item.get(ph_key, ph_def)
-            ph = item.get('pos_hint', ph)
-
-            sh = sh_def if sh_key is None else item.get(sh_key, sh_def)
-            sh = item.get('size_hint', sh)
-            sh = [item.get('size_hint_x', sh[0]),
-                  item.get('size_hint_y', sh[1])]
-
-            sh_min = sh_min_def if sh_min_key is None else item.get(sh_min_key,
-                                                                    sh_min_def)
-            sh_min = item.get('size_hint_min', sh_min)
-            sh_min = [item.get('size_hint_min_x', sh_min[0]),
-                      item.get('size_hint_min_y', sh_min[1])]
-
-            sh_max = sh_max_def if sh_max_key is None else item.get(sh_max_key,
-                                                                    sh_max_def)
-            sh_max = item.get('size_hint_max', sh_max)
-            sh_max = [item.get('size_hint_max_x', sh_max[0]),
-                      item.get('size_hint_max_y', sh_max[1])]
-
-            s = s_def if s_key is None else item.get(s_key, s_def)
-            s = item.get('size', s)
-            w, h = s = item.get('width', s[0]), item.get('height', s[1])
-
-            viewcls = None
-            if viewcls_key is not None:
-                viewcls = item.get(viewcls_key)
-                if viewcls is not None:
-                    viewcls = getattr(Factory, viewcls)
-            if viewcls is None:
-                viewcls = viewcls_def
-
-            opts[i] = {
-                'size': [(iw if w is None else w), (ih if h is None else h)],
-                'size_hint': sh, 'size_hint_min': sh_min,
-                'size_hint_max': sh_max, 'pos': None, 'pos_hint': ph,
-                'viewclass': viewcls, 'width_none': w is None,
-                'height_none': h is None}
+        ph = ph_def if ph_key is None else item.get(ph_key, ph_def)
+        ph = item.get('pos_hint', ph)
+
+        sh = sh_def if sh_key is None else item.get(sh_key, sh_def)
+        sh = item.get('size_hint', sh)
+        sh = [item.get('size_hint_x', sh[0]),
+              item.get('size_hint_y', sh[1])]
+
+        sh_min = sh_min_def if sh_min_key is None else item.get(sh_min_key,
+                                                                sh_min_def)
+        sh_min = item.get('size_hint_min', sh_min)
+        sh_min = [item.get('size_hint_min_x', sh_min[0]),
+                  item.get('size_hint_min_y', sh_min[1])]
+
+        sh_max = sh_max_def if sh_max_key is None else item.get(sh_max_key,
+                                                                sh_max_def)
+        sh_max = item.get('size_hint_max', sh_max)
+        sh_max = [item.get('size_hint_max_x', sh_max[0]),
+                  item.get('size_hint_max_y', sh_max[1])]
+
+        s = s_def if s_key is None else item.get(s_key, s_def)
+        s = item.get('size', s)
+        w, h = s = item.get('width', s[0]), item.get('height', s[1])
+
+        viewcls = None
+        if viewcls_key is not None:
+            viewcls = item.get(viewcls_key)
+            if viewcls is not None:
+                viewcls = getattr(Factory, viewcls)
+        if viewcls is None:
+            viewcls = viewcls_def
+
+        return {
+            'size': [(iw if w is None else w), (ih if h is None else h)],
+            'size_hint': sh, 'size_hint_min': sh_min,
+            'size_hint_max': sh_max, 'pos': None, 'pos_hint': ph,
+            'viewclass': viewcls, 'width_none': w is None,
+            'height_none': h is None}
+
+    def converted_opts(self):
+        return [opt if opt is not None else self._convert_opt({}) for opt in self.view_opts]

     def compute_layout(self, data, flags):
         self._size_needs_update = False
@@ -331,6 +335,8 @@ class RecycleLayout(RecycleLayoutManagerBehavior, Layout):
         changed = []
         for widget, index in self.view_indices.items():
             opt = opts[index]
+            if opt is None:
+                opt = self._convert_opt({})
             s = opt['size']
             w, h = sn = list(widget.size)
             sh = opt['size_hint']
@@ -366,7 +372,7 @@ class RecycleLayout(RecycleLayoutManagerBehavior, Layout):
         assert False

     def set_visible_views(self, indices, data, viewport):
-        view_opts = self.view_opts
+        view_opts = self.converted_opts()
         new, remaining, old = self.recycleview.view_adapter.set_visible_views(
             indices, data, view_opts)

@@ -414,7 +420,7 @@ class RecycleLayout(RecycleLayoutManagerBehavior, Layout):
             self.recycleview.refresh_from_layout(view_size=True)

     def refresh_view_layout(self, index, layout, view, viewport):
-        opt = self.view_opts[index].copy()
+        opt = self._convert_opt(self.view_opts.get(index, {}))
         width_none = opt.pop('width_none')
         height_none = opt.pop('height_none')
         opt.update(layout)
EOF
# compile kivy
RUN <<EOF
	set -eux;
	cd kivy-*;
    KIVY_DEPS_ROOT="$PWD/kivy-dependencies";
    echo "KIVY_DEPS_ROOT=$KIVY_DEPS_ROOT";
    export KIVY_DEPS_ROOT;
	for minor in $(seq 12 14); do
		python3.$minor -m pip install --root-user-action ignore Cython tomli-w u-msgpack-python sortedcontainers zict typing-extensions tornado toolz toml tblib soupsieve six pyyaml python-dotenv pyparsing pyarrow psutil ppft pox pluggy platformdirs pillow packaging numpy networkx msgpack more-itertools MarkupSafe lxml locket kiwisolver iniconfig greenlet fsspec fonttools dill cycler cloudpickle click blinker attrs annotated-types variconfig sqlalchemy python-dateutil pytest partd multiprocess jinja2 contourpy beautifulsoup4 pathos pandas matplotlib dask distributed parquetdb;
		USE_SDL3=1 python3.$minor -m pip wheel .;
		python3.$minor -m pip install --root-user-action ignore kivy*-cp3$minor-linux_x86_64.whl;
	done;
	cd ..;
EOF
# make some useful symlinks that are expected to exist ("/usr/local/bin/python" and friends)
RUN set -eux; \
	for src in idle3 pip3 pydoc3 python3 python3-config; do \
		dst="$(echo "$src" | tr -d 3)"; \
		[ -s "/usr/local/bin/$src" ]; \
		[ ! -e "/usr/local/bin/$dst" ]; \
		ln -svT "$src" "/usr/local/bin/$dst"; \
	done
