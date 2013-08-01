-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO menu (name, bottom, left, right, top, style) VALUES
('Main', 0.12, 0.0, 0.1, 1.0, 'BigLight');
INSERT INTO menu (name) VALUES ('Game'), ('Editor');
INSERT INTO menu_item (idx, menu, closer, on_click, text) VALUES
(0, 'Game', 1, 'start_new_map()', '@new_map'),
(1, 'Game', 1, 'open_map()', '@open_map'),
(2, 'Game', 1, 'quit_map_editor()', '@quit_maped'),
(3, 'Game', 1, 'save_map()', '@save_map'),
(0, 'Editor', 1, 'editor_select()', '@ed_select'),
(1, 'Editor', 1, 'editor_copy()', '@ed_copy'),
(2, 'Editor', 1, 'editor_paste()', '@ed_paste'),
(3, 'Editor', 1, 'editor_delete()', '@ed_delete'),
(0, 'Main', 0, 'mi_create_thing()', '@thing_menu'),
(1, 'Main', 0, 'mi_create_place()', '@place_menu'),
(2, 'Main', 0, 'mi_create_portal()', '@portal_menu'),
(3, 'Main', 0, 'toggle_menu(Game)', '@game_menu'),
(4, 'Main', 0, 'toggle_menu(Editor)', '@editor_menu');
