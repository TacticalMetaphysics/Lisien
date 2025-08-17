import pytest
from kivy.resources import resource_add_path, resource_find
import kivy.tests

import elide  # may add resource paths at import time


def test_elide_dot_kv(elide_app):
	assert resource_find("elide.kv")
