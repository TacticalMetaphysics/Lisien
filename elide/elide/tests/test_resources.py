from kivy.resources import resource_find

import elide  # may add resource paths at import time


def test_elide_dot_kv():
	assert resource_find("elide.kv")
