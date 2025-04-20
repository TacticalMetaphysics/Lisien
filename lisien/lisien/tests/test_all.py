# This file is part of allegedb, an object-relational mapper for versioned graphs.
# Copyright (c) Zachary Spector. public@zacharyspector.com
import unittest
from copy import deepcopy
from types import SimpleNamespace

import pytest

from lisien.engine import Engine

testkvs = [
	0,
	1,
	10,
	10**10,
	"spam",
	"eggs",
	"ham",
	"💧",
	"🔑",
	"𐦖",
	("spam", "eggs", "ham"),
]
testvs = [["spam", "eggs", "ham"], {"foo": "bar", 0: 1, "💧": "🔑"}]
testdata = []
for k in testkvs:
	for v in testkvs:
		testdata.append((k, v))
	for v in testvs:
		testdata.append((k, v))
testdata.append(("lol", deepcopy(testdata)))


class AllegedTest(unittest.TestCase):
	def setUp(self):
		self.engine = Engine(
			None,
		)
		self.graphmakers = (self.engine.new_character,)


class AbstractGraphTest:
	def test_graph_objects_create_delete(self):
		g = self.engine.new_character("physical")
		self.assertFalse(self.engine._node_exists("physical", 0))
		g.add_node(0)
		self.assertTrue(self.engine._node_exists("physical", 0))
		self.assertIn(0, g)
		g.add_node(1)
		self.assertIn(1, g)
		g.add_edge(0, 1)
		self.assertIn(1, g.adj[0])
		self.assertIn(1, list(g.adj[0]))
		g.add_edge(2, 3)
		self.assertIn(2, g.node)
		self.assertIn(3, g.node)
		self.assertIn(2, g.adj)
		self.assertIn(3, g.adj[2])
		self.assertIn(3, list(g.adj[2]))
		if hasattr(g, "pred_cls"):
			self.assertIn(2, g.pred[3])
			g.add_edge(2, 4)
			self.assertIn(2, g.pred[4])
			self.assertIn(2, list(g.pred[4]))
			self.assertIn(4, g.adj[2])
			self.assertIn(4, list(g.adj[2]))
			del g.pred[4]
			self.assertEqual(len(g.pred[4]), 0)
			self.assertNotIn(4, g.adj[2])
			self.assertNotIn(4, list(g.adj[2]))
			self.assertIn(4, g.node)
			self.assertNotIn(0, g.adj[1])
			self.assertNotIn(0, list(g.adj[1]))
		else:
			self.assertIn(0, g.adj[1])
			self.assertIn(0, list(g.adj[1]))
		self.engine.next_turn()
		self.assertIn(0, g)
		self.assertIn(1, g)
		self.engine.branch = "physical_no_edge"
		self.assertIn(3, g.node)
		self.assertIn(0, g)
		self.assertTrue(self.engine._node_exists("physical", 0))
		self.assertIn(1, g)
		self.assertIn(1, g.adj[0])
		self.assertIn(1, list(g.adj[0]))
		if hasattr(g, "pred_cls"):
			self.assertNotIn(0, g.adj[1])
			self.assertNotIn(0, list(g.adj[1]))
		else:
			self.assertIn(0, g.adj[1])
			self.assertIn(0, list(g.adj[1]))
		g.remove_edge(0, 1)
		self.assertIn(0, g)
		self.assertIn(1, g)
		self.assertNotIn(1, g.adj[0])
		self.assertNotIn(1, list(g.adj[0]))
		self.assertIn(0, g.adj)
		self.assertNotIn(1, g.adj[0])
		self.engine.branch = "physical_triangle"
		self.assertIn(3, g.node)
		self.assertIn(2, g)
		g.add_edge(0, 1)
		self.assertIn(1, g.adj[0])
		self.assertIn(1, list(g.adj[0]))
		if g.is_directed():
			g.add_edge(1, 0)
		self.assertIn(0, g.adj[1])
		self.assertIn(0, list(g.adj[1]))
		g.add_edge(1, 2)
		g.add_edge(2, 1)
		g.add_edge(2, 0)
		g.add_edge(0, 2)
		self.assertIn(2, g.adj[0])
		self.assertIn(2, list(g.adj[0]))
		self.engine.branch = "physical_square"
		self.assertTrue(self.engine._node_exists("physical", 0))
		self.assertIn(3, g.node)
		self.assertIn(2, list(g.adj[0]))
		self.assertIn(2, g.adj[0])
		self.engine.next_turn()
		self.assertIn(2, g)
		self.assertIn(2, list(g.node.keys()))
		self.assertIn(2, list(g.adj[0]))
		self.assertIn(2, g.adj[0])
		self.assertIn(2, list(g.adj[0]))
		self.assertTrue(self.engine._node_exists("physical", 0))
		g.remove_edge(2, 0)
		self.assertNotIn(0, g.adj[2])
		self.assertNotIn(0, list(g.adj[2]))
		self.assertIn(0, g.node)
		self.assertTrue(self.engine._node_exists("physical", 0))
		self.assertNotIn(0, g.adj[3])
		g.add_edge(3, 0)
		self.assertIn(0, g.adj[3])
		self.assertEqual(self.engine.turn, 2)
		self.assertIn(0, g.adj[3])
		self.assertIn(0, list(g.adj[3]))
		self.assertIn(0, g.node)
		self.assertTrue(self.engine._node_exists("physical", 0))
		if g.is_directed():
			self.assertIn(2, g.pred[3])
			self.assertIn(3, g.pred[0])
		self.engine.branch = "physical_de_edge"
		self.assertIn(3, g.node)
		self.assertIn(0, g.node)
		self.assertTrue(self.engine._node_exists("physical", 0))
		g.remove_node(3)
		self.assertNotIn(3, g.node)
		self.assertNotIn(3, g.adj)
		self.assertNotIn(3, g.adj[2])
		if g.is_directed():
			self.assertNotIn(3, g.pred)
			self.assertNotIn(3, g.pred[0])
		self.engine.branch = "physical_square"
		self.assertEqual(self.engine.turn, 2)
		self.assertNotIn(0, g.adj[2])
		self.assertNotIn(0, list(g.adj[2]))
		self.assertIn(0, g.adj[3])
		self.assertIn(0, list(g.adj[3]))
		self.assertIn(3, g.node)
		self.engine.branch = "physical_nothing"
		self.assertNotIn(0, g.adj[2])
		self.assertNotIn(0, list(g.adj[2]))
		self.assertIn(0, g.adj[3])
		self.assertIn(0, list(g.adj[3]))
		self.assertIn(3, g.node)
		g.remove_nodes_from((0, 1, 2, 3))
		for n in (0, 1, 2, 3):
			self.assertNotIn(n, g.node)
			self.assertNotIn(n, g.adj)


class AbstractBranchLineageTest(AbstractGraphTest):
	# TODO: an analogue of this test for when you're looking up keyframes
	#       in parent branches
	def test_graph_objects_create_delete(self):
		"""Create some branches of history and check that allegedb remembers where
		each came from and what happened in each.

		"""
		super().test_graph_objects_create_delete()
		gmn = "physical"
		self.assertTrue(self.engine.is_ancestor_of("trunk", gmn + "_no_edge"))
		self.assertTrue(self.engine.is_ancestor_of("trunk", gmn + "_triangle"))
		self.assertTrue(self.engine.is_ancestor_of("trunk", gmn + "_nothing"))
		self.assertTrue(
			self.engine.is_ancestor_of(gmn + "_no_edge", gmn + "_triangle")
		)
		self.assertTrue(
			self.engine.is_ancestor_of(gmn + "_square", gmn + "_nothing")
		)
		self.assertFalse(self.engine.is_ancestor_of(gmn + "_nothing", "trunk"))
		self.assertFalse(
			self.engine.is_ancestor_of(gmn + "_triangle", gmn + "_no_edge")
		)
		self.engine.turn = self.engine._branch_start("trunk")[0]
		self.engine.branch = "trunk"
		g = self.engine.character[gmn]
		self.assertIn(0, g.node)
		self.assertIn(1, g.node)
		self.assertIn(0, g.edge)
		self.assertIn(1, g.edge[0])
		self.engine.turn = 0

		def badjump():
			self.engine.branch = gmn + "_no_edge"

		self.assertRaises(ValueError, badjump)
		self.engine.turn = self.engine.branch_start_turn(gmn + "_no_edge")
		self.engine.branch = gmn + "_no_edge"
		self.engine.next_turn()
		self.assertIn(0, g)
		self.assertIn(0, list(g.node.keys()))
		self.assertNotIn(1, g.edge[0])
		self.assertNotIn(0, g.edge[1])
		if g.is_multigraph():
			self.assertRaises(KeyError, lambda: g.edge[0][1][0])
		else:
			self.assertRaises(KeyError, lambda: g.edge[0][1])
		self.engine.branch = gmn + "_triangle"
		self.assertIn(2, g.node)
		for orig in (0, 1, 2):
			for dest in (0, 1, 2):
				if orig == dest:
					continue
				self.assertIn(orig, g.edge)
				self.assertIn(dest, g.edge[orig])
		self.engine.branch = gmn + "_square"
		self.assertNotIn(0, g.edge[2])
		if g.is_multigraph():
			self.assertRaises(KeyError, lambda: g.edge[2][0][0])
		else:
			self.assertRaises(KeyError, lambda: g.edge[2][0])
		self.engine.turn = 2
		self.assertIn(3, g.node)
		self.assertIn(1, g.edge[0])
		self.assertIn(2, g.edge[1])
		self.assertIn(3, g.edge[2])
		self.assertIn(0, g.edge[3])
		self.engine.branch = gmn + "_nothing"
		for node in (0, 1, 2):
			self.assertNotIn(node, g.node)
			self.assertNotIn(node, g.edge)
		self.engine.branch = "trunk"
		self.engine.turn = 0
		self.assertIn(0, g.node)
		self.assertIn(1, g.node)
		self.assertIn(0, g.edge)
		self.assertIn(1, g.edge[0])


class BranchLineageTest(AbstractBranchLineageTest, AllegedTest):
	pass


class StorageTest(AllegedTest):
	def test_store_value(self):
		"""Test that all the graph types can store and retrieve key-value pairs
		for the graph as a whole, for nodes, and for edges.

		"""
		g = self.engine.new_character("testgraph")
		g.add_node(0)
		g.add_node(1)
		g.add_edge(0, 1)
		n = g.node[0]
		e = g.edge[0][1]
		for k, v in testdata:
			g.graph[k] = v
			self.assertIn(k, g.graph)
			self.assertEqual(g.graph[k], v)
			del g.graph[k]
			self.assertNotIn(k, g.graph)
			n[k] = v
			self.assertIn(k, n)
			self.assertEqual(n[k], v)
			del n[k]
			self.assertNotIn(k, n)
			e[k] = v
			self.assertIn(k, e)
			self.assertEqual(e[k], v)
			del e[k]
			self.assertNotIn(k, e)
		self.engine.del_character("testgraph")

	def test_store_dict(self):
		for i, graphmaker in enumerate(self.graphmakers):
			self.engine.turn = i
			g = graphmaker("testgraph")
			g.add_node(0)
			g.add_node(1)
			g.add_edge(0, 1)
			n = g.node[0]
			e = g.edge[0][1]
			for entity in g.graph, n, e:
				entity[0] = {
					"spam": "eggs",
					"ham": {"baked beans": "delicious"},
					"qux": ["quux", "quuux"],
					"clothes": {"hats", "shirts", "pants"},
					"dicts": {"foo": {"bar": "bas"}, "qux": {"quux": "quuux"}},
				}
			self.engine.next_turn()
			for entity in g.graph, n, e:
				self.assertEqual(entity[0]["spam"], "eggs")
				entity[0]["spam"] = "ham"
				self.assertEqual(entity[0]["spam"], "ham")
				self.assertEqual(
					entity[0]["ham"], {"baked beans": "delicious"}
				)
				entity[0]["ham"]["baked beans"] = "disgusting"
				self.assertEqual(
					entity[0]["ham"], {"baked beans": "disgusting"}
				)
				self.assertEqual(entity[0]["qux"], ["quux", "quuux"])
				entity[0]["qux"] = ["quuux", "quux"]
				self.assertEqual(entity[0]["qux"], ["quuux", "quux"])
				self.assertEqual(
					entity[0]["clothes"], {"hats", "shirts", "pants"}
				)
				entity[0]["clothes"].remove("hats")
				self.assertEqual(entity[0]["clothes"], {"shirts", "pants"})
				self.assertEqual(
					entity[0]["dicts"],
					{"foo": {"bar": "bas"}, "qux": {"quux": "quuux"}},
				)
				del entity[0]["dicts"]["foo"]
				entity[0]["dicts"]["qux"]["foo"] = {"bar": "bas"}
				self.assertEqual(
					entity[0]["dicts"],
					{"qux": {"foo": {"bar": "bas"}, "quux": "quuux"}},
				)
			self.engine.turn = i
			for entity in g.graph, n, e:
				self.assertEqual(entity[0]["spam"], "eggs")
				self.assertEqual(
					entity[0]["ham"], {"baked beans": "delicious"}
				)
				self.assertEqual(entity[0]["qux"], ["quux", "quuux"])
				self.assertEqual(
					entity[0]["clothes"], {"hats", "shirts", "pants"}
				)
				self.assertEqual(
					entity[0]["dicts"],
					{"foo": {"bar": "bas"}, "qux": {"quux": "quuux"}},
				)

	def test_store_list(self):
		for i, graphmaker in enumerate(self.graphmakers):
			self.engine.turn = i
			g = graphmaker("testgraph")
			g.add_node(0)
			g.add_node(1)
			g.add_edge(0, 1)
			n = g.node[0]
			e = g.edge[0][1]
			for entity in g.graph, n, e:
				entity[0] = [
					"spam",
					("eggs", "ham"),
					{"baked beans": "delicious"},
					["qux", "quux", "quuux"],
					{"hats", "shirts", "pants"},
				]
			self.engine.next_turn()
			for entity in g.graph, n, e:
				self.assertEqual(entity[0][0], "spam")
				entity[0][0] = "eggplant"
				self.assertEqual(entity[0][0], "eggplant")
				self.assertEqual(entity[0][1], ("eggs", "ham"))
				entity[0][1] = ("ham", "eggs")
				self.assertEqual(entity[0][1], ("ham", "eggs"))
				self.assertEqual(entity[0][2], {"baked beans": "delicious"})
				entity[0][2]["refried beans"] = "deliciouser"
				self.assertEqual(
					entity[0][2],
					{
						"baked beans": "delicious",
						"refried beans": "deliciouser",
					},
				)
				self.assertEqual(entity[0][3], ["qux", "quux", "quuux"])
				entity[0][3].pop()
				self.assertEqual(entity[0][3], ["qux", "quux"])
				self.assertEqual(entity[0][4], {"hats", "shirts", "pants"})
				entity[0][4].discard("shame")
				entity[0][4].remove("pants")
				entity[0][4].add("sun")
				self.assertEqual(entity[0][4], {"hats", "shirts", "sun"})
			self.engine.turn = i
			for entity in g.graph, n, e:
				self.assertEqual(entity[0][0], "spam")
				self.assertEqual(entity[0][1], ("eggs", "ham"))
				self.assertEqual(entity[0][2], {"baked beans": "delicious"})
				self.assertEqual(entity[0][3], ["qux", "quux", "quuux"])
				self.assertEqual(entity[0][4], {"hats", "shirts", "pants"})

	def test_store_set(self):
		for i, graphmaker in enumerate(self.graphmakers):
			self.engine.turn = i
			g = graphmaker("testgraph")
			g.add_node(0)
			g.add_node(1)
			g.add_edge(0, 1)
			n = g.node[0]
			e = g.edge[0][1]
			for entity in g.graph, n, e:
				entity[0] = set(range(10))
			self.engine.next_turn()
			for entity in g.graph, n, e:
				self.assertEqual(entity[0], set(range(10)))
				for j in range(0, 12, 2):
					entity[0].discard(j)
				self.assertEqual(entity[0], {1, 3, 5, 7, 9})
			self.engine.turn = i
			for entity in g.graph, n, e:
				self.assertEqual(entity[0], set(range(10)))


if __name__ == "__main__":
	unittest.main()
