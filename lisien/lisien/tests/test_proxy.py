# This file is part of Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import shutil
import tempfile
from unittest.mock import patch

import networkx as nx
import pytest

import lisien.examples.kobold as kobold
import lisien.examples.polygons as polygons
import lisien.tests.test_all
from lisien.engine import Engine
from lisien.proxy import EngineProcessManager
from lisien.proxy.handle import EngineHandle
from lisien.tests import data


class ProxyTest(lisien.tests.test_all.AllegedTest):
	def setUp(self):
		self.manager = EngineProcessManager()
		self.tmp_path = tempfile.mkdtemp()
		self.engine = self.manager.start(
			self.tmp_path,
			connect_string="sqlite:///:memory:",
			enforce_end_of_time=False,
			workers=0,
		)
		self.graphmakers = (self.engine.new_character,)
		self.addCleanup(self._do_cleanup)

	def _do_cleanup(self):
		self.manager.shutdown()
		shutil.rmtree(self.tmp_path)


class ProxyGraphTest(lisien.tests.test_all.AbstractGraphTest, ProxyTest):
	pass


class ProxyStorageTest(ProxyTest, lisien.tests.test_all.StorageTest):
	pass


def test_fast_delta(handle_initialized):
	hand = handle_initialized
	unpack_delta = hand._real._unpack_slightly_packed_delta

	# there's currently no way to do fast delta past the time when
	# a character was created, due to the way keyframes work...
	# so don't test that

	branch, turn, tick = hand._real._btt()
	ret, diff = hand.next_turn()
	btt = hand._real._btt()
	slowd = unpack_delta(
		hand._get_slow_delta(btt_from=(branch, turn, tick), btt_to=btt)
	)
	fastd = hand.unpack(diff)
	if "universal" in slowd:
		del slowd["universal"]["rando_state"]
		del fastd["universal"]["rando_state"]
	assert fastd == slowd, "Fast delta differs from slow delta"
	ret, diff2 = hand.time_travel("trunk", 0, tick)
	btt2 = hand._real._btt()
	slowd2 = unpack_delta(hand._get_slow_delta(btt_from=btt, btt_to=btt2))
	fastd2 = hand.unpack(diff2)
	if "universal" in slowd:
		del slowd2["universal"]["rando_state"]
		del fastd2["universal"]["rando_state"]
	assert fastd2 == slowd2, "Fast delta differs from slow delta"
	ret, diff3 = hand.time_travel("trunk", 1)
	btt3 = hand._real._btt()
	slowd3 = unpack_delta(hand._get_slow_delta(btt_from=btt2, btt_to=btt3))
	fastd3 = hand.unpack(diff3)
	if "universal" in slowd:
		del slowd3["universal"]["rando_state"]
		del fastd3["universal"]["rando_state"]
	assert fastd3 == slowd3, "Fast delta differs from slow delta"


def test_serialize_deleted(college24_premade):
	eng = college24_premade
	d0r0s0 = eng.character["dorm0room0student0"]
	roommate = d0r0s0.stat["roommate"]
	del eng.character[roommate.name]
	assert not roommate
	with pytest.raises(KeyError):
		eng.character[roommate.name]
	assert d0r0s0.stat["roommate"] == roommate
	assert eng.unpack(eng.pack(d0r0s0.stat["roommate"])) == roommate


def test_manip_deleted(sqleng):
	eng = sqleng
	phys = eng.new_character("physical")
	phys.stat["aoeu"] = True
	phys.add_node(0)
	phys.add_node(1)
	phys.node[1]["aoeu"] = True
	del phys.node[1]
	phys.add_node(1)
	assert "aoeu" not in phys.node[1]
	phys.add_edge(0, 1)
	phys.adj[0][1]["aoeu"] = True
	del phys.adj[0][1]
	phys.add_edge(0, 1)
	assert "aoeu" not in phys.adj[0][1]
	del eng.character["physical"]
	assert not phys
	phys = eng.new_character("physical")
	assert "aoeu" not in phys.stat
	assert 0 not in phys
	assert 1 not in phys
	assert 0 not in phys.adj
	assert 1 not in phys.adj


class TestSwitchMainBranch(ProxyTest):
	def test_switch_main_branch(self):
		phys = self.engine.new_character("physical", hello="hi")
		self.engine.next_turn()
		phys.stat["hi"] = "hello"
		with pytest.raises(ValueError):
			self.engine.switch_main_branch("tronc")
		self.engine.turn = 0
		self.engine.tick = 0
		self.engine.switch_main_branch("tronc")
		assert self.engine.branch == "tronc"
		assert phys
		assert "hi" not in phys.stat
		assert "hello" in phys.stat
		self.engine.next_turn()
		phys.stat["hi"] = "hey there"
		self.engine.turn = 0
		self.engine.tick = 0
		self.engine.switch_main_branch("trunk")
		assert phys.stat["hello"] == "hi"
		self.engine.turn = 1
		assert phys.stat["hello"] == "hi"
		assert phys.stat["hi"] == "hello"


def test_updnoderb(handle):
	engine = handle._real
	char0 = engine.new_character("0")
	node0 = char0.new_place("0")

	@node0.rule(always=True)
	def change_rulebook(node):
		node.rulebook = "haha"

	a, b = handle.next_turn()

	delta = engine.unpack(b)

	assert delta
	assert "0" in delta
	assert "node_val" in delta["0"]
	assert "0" in delta["0"]["node_val"]
	assert "0" in delta["0"]["node_val"]
	assert "rulebook" in delta["0"]["node_val"]["0"]
	assert delta["0"]["node_val"]["0"]["rulebook"] == "haha"


def test_updedgerb(handle):
	engine = handle._real
	char0 = engine.new_character("0")
	node0 = char0.new_place("0")
	node1 = char0.new_place("1")
	edge = node0.new_portal(node1)

	@edge.rule(always=True)
	def change_rulebook(edge):
		edge.rulebook = "haha"

	a, b = handle.next_turn()

	delta = engine.unpack(b)

	assert (
		"0" in delta
		and "edge_val" in delta["0"]
		and "0" in delta["0"]["edge_val"]
		and "1" in delta["0"]["edge_val"]["0"]
		and "rulebook" in delta["0"]["edge_val"]["0"]["1"]
		and delta["0"]["edge_val"]["0"]["1"]["rulebook"] == "haha"
	)


def test_thing_place_iter(tmp_path):
	# set up some world state with things and places, before starting the proxy
	with lisien.Engine(tmp_path, workers=0) as eng:
		kobold.inittest(eng)
	manager = EngineProcessManager()
	engine = manager.start(tmp_path, workers=0)
	phys = engine.character["physical"]
	for place_name in phys.place:
		assert isinstance(place_name, tuple)
	for thing_name in phys.thing:
		assert isinstance(thing_name, str)
	manager.shutdown()


@pytest.fixture
def mocked_keyframe(tmp_path):
	with (
		patch("lisien.Engine.snap_keyframe"),
		Engine(
			tmp_path,
			random_seed=69105,
			enforce_end_of_time=False,
			keyframe_on_close=False,
			workers=0,
		) as eng,
	):
		for _ in range(8):
			eng.next_turn()
		eng._start_branch("trunk", *data.BTT_FROM)
		eng.snap_keyframe.side_effect = [data.KF_FROM, data.KF_TO]
		eng._set_btt(*data.BTT_FROM)
		yield eng


@pytest.mark.parametrize("run", list(range(10)))
def test_get_slow_delta_overload(mocked_keyframe, run):
	eng = mocked_keyframe
	slowd = eng._unpack_slightly_packed_delta(
		eng._get_slow_delta(data.BTT_FROM, data.BTT_TO)
	)
	assert slowd == data.SLOW_DELTA


@pytest.mark.parametrize("algorithm", ["slow", "fast"])
def test_apply_delta(tmp_path, algorithm):
	slow = algorithm == "slow"
	with Engine(tmp_path, workers=0) as eng:
		initial_state = nx.DiGraph(
			{
				0: {1: {"omg": "lol"}},
				1: {0: {"omg": "blasphemy"}},
				2: {},
				3: {},
				"it": {},
			}
		)
		initial_state.nodes()[2]["hi"] = "hello"
		initial_state.nodes()["it"]["location"] = 0
		initial_state.graph["wat"] = "nope"
		phys = eng.new_character("physical", initial_state)
		eng.add_character("pointless")
		kf0 = eng.snap_keyframe()
		del kf0["universal"]["rando_state"]
		if slow:
			eng.branch = "b"
		else:
			eng.next_turn()
		kf1 = eng.snap_keyframe()
		del kf1["universal"]["rando_state"]
		assert kf0 == kf1
		del phys.portal[1][0]
		port = phys.new_portal(0, 2)
		port["hi"] = "bye"
		phys.place[1]["wtf"] = "bbq"
		phys.thing["it"].location = phys.place[1]
		del phys.place[3]
		eng.add_character("pointed")
		del eng.character["pointless"]
		assert "pointless" not in eng.character, "Failed to delete character"
		phys.portal[0][1]["meaning"] = 42
		del phys.portal[0][1]["omg"]
		if slow:
			eng.branch = "trunk"
		else:
			eng.turn = 0
			eng.tick = 0
	mang = EngineProcessManager()
	try:
		prox = mang.start(tmp_path, workers=0)
		assert prox.turn == 0
		phys = prox.character["physical"]
		assert 3 in phys.place
		assert phys.portal[1][0]["omg"] == "blasphemy"
		assert "it" in phys.thing
		if slow:
			prox.branch = "b"
		else:
			prox.turn = 1
		assert 3 not in phys.place
		assert not list(phys.portal[1])
		assert 2 in phys.portal[0]
		assert phys.portal[0][2]["hi"] == "bye"
		assert phys.place[1]["wtf"] == "bbq"
		assert phys.thing["it"].location == phys.place[1]
		assert "pointless" not in prox.character, "Loaded deleted character"
		assert "pointed" in prox.character
		assert phys.portal[0][1]["meaning"] == 42
		assert "omg" not in phys.portal[0][1]
	finally:
		mang.shutdown()


@pytest.fixture
def polys(tmp_path):
	with Engine(tmp_path, workers=0, random_seed=69105) as eng:
		polygons.install(eng)
	return tmp_path


def test_change_triggers(polys):
	procman = EngineProcessManager()
	eng = procman.start(polys)
	relocate = eng.character["triangle"].unit.rule["relocate"]
	assert list(relocate.triggers) == [
		eng.trigger.similar_neighbors,
		eng.trigger.dissimilar_neighbors,
	]
	relocate.triggers = ["dissimilar_neighbors"]
	procman.shutdown()
	with Engine(polys, workers=0) as eng:
		assert list(
			eng.character["triangle"].unit.rule["relocate"].triggers
		) == [eng.trigger.dissimilar_neighbors]


def test_change_string(tmp_path):
	handle = EngineHandle(tmp_path, workers=0)
	handle.set_language("eng")
	handle.set_string("a string", "its value")
	assert handle.get_string_lang_items("eng") == [("a string", "its value")]
	handle.close()
	with Engine(tmp_path, workers=0) as eng:
		assert eng.string["a string"] == "its value"
