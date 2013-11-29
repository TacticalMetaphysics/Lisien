-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO spot_coords (place, x, y) VALUES
('myroom', 400, 100),
('mybathroom', 450, 150),
('guestroom', 400, 200),
('livingroom', 300, 150),
('diningoffice', 350, 200),
('kitchen', 350, 150),
('longhall', 250, 150),
('momsroom', 250, 100),
('momsbathroom', 250, 200),
('outside', 300, 100);
INSERT INTO spot_interactive (place) SELECT place FROM spot_coords;
INSERT INTO spot_img (place) SELECT place FROM spot_coords;
