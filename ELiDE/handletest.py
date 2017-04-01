from LiSE.handle import EngineHandle

hand = EngineHandle([':memory:'])
hand.add_character('physical', {}, {})
hand.set_place('physical', 'a', {'evil': False})
assert not hand._node_stat_cache['physical']['a']['evil']
assert not hand.get_node_stat('physical', 'a', 'evil')
hand.time_travel('trunk', 1)
assert hand._real.branch == 'trunk'
assert hand._real.tick == hand._real.rev == 1
assert not hand.get_node_stat('physical', 'a', 'evil')
assert not hand._node_stat_cache['physical']['a']['evil']
hand.set_node_stat('physical', 'a', 'evil', True)
assert hand._node_stat_cache['physical']['a']['evil']
assert hand.get_node_stat('physical', 'a', 'evil')
olde = hand.node_stat_copy('physical', 'a')
dyff = hand.time_travel('trunk', 0)
neue = hand.node_stat_copy('physical', 'a')
assert olde != neue
assert not hand.get_node_stat('physical', 'a', 'evil')
assert not hand._node_stat_cache['physical']['a']['evil']
assert not dyff['physical']['node_stat']['a']['evil']

hand.close()
