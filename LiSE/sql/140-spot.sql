-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO spot (place) VALUES
('zack_room'),
('zack_bathroom'),
('guestroom'),
('livingroom'),
('diningoffice'),
('kitchen'),
('longhall'),
('gail_room'),
('gail_bathroom'),
('balcony'),
('apt_hall');

INSERT INTO spot_coords (place, x, y) VALUES
('zack_room', 400, 100),
('zack_bathroom', 450, 150),
('guestroom', 400, 200),
('livingroom', 300, 150),
('diningoffice', 350, 200),
('kitchen', 350, 150),
('longhall', 250, 150),
('gail_room', 250, 100),
('gail_bathroom', 250, 200),
('balcony', 300, 100),
('apt_hall', 300, 400);
