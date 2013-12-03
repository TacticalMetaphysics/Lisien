-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO menu_item (idx, menu, closer, on_click, text) VALUES
(0, 'Game', 1, 'start_new_map()', '@new_map'),
(1, 'Game', 1, 'open_map()', '@open_map'),
(2, 'Game', 1, 'quit_map_editor()', '@quit_maped'),
(3, 'Game', 1, 'save_map()', '@save_map'),
(0, 'Editor', 1, 'editor_select()', '@ed_select'),
(1, 'Editor', 1, 'editor_copy()', '@ed_copy'),
(2, 'Editor', 1, 'editor_paste()', '@ed_paste'),
(3, 'Editor', 1, 'editor_delete()', '@ed_delete'),
(0, 'Main', 0, 'mi_show_popup(new_thing(base, body))', '@thing_menu'),
(1, 'Main', 0, 'mi_show_popup(new_place(!spot))', '@place_menu'),
(2, 'Main', 0, 'mi_connect_portal()', '@portal_menu'),
(3, 'Main', 0, 'toggle_menu(Game)', '@game_menu'),
(4, 'Main', 0, 'toggle_menu(Editor)', '@editor_menu'),
(13, 'Main', 0, 'noop()', 'Branch:'),
(14, 'Main', 0, 'noop()', '@branch'),
(15, 'Main', 0, 'noop()', 'Tick:'),
(16, 'Main', 0, 'noop()', '@tick');
INSERT INTO menu_item (idx, menu, on_click, text, symbolic) VALUES
(5, 'Main', 'stop()', '@pause', 1),
(6, 'Main', 'play_speed(1)', '@forward', 1),
(7, 'Main', 'play_speed(-1)', '@reverse', 1),
(8, 'Main', 'back_to_start()', '@beginning', 1),
(9, 'Main', 'time_travel_inc_tick(1)', '@stepforward', 1),
(10, 'Main', 'time_travel_inc_tick(-1)', '@stepbackward', 1),
(11, 'Main', 'time_travel_inc_branch(1)', '@next_branch', 1),
(12, 'Main', 'time_travel_inc_branch(-1)', '@prev_branch', 1);
