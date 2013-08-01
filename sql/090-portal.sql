-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO portal_existence (origin, destination) VALUES
('myroom', 'guestroom'),
        ('myroom', 'mybathroom'),
        ('myroom', 'diningoffice'),
        ('myroom', 'livingroom'),
        ('guestroom', 'diningoffice'),
        ('guestroom', 'livingroom'),
        ('guestroom', 'mybathroom'),
        ('livingroom', 'diningoffice'),
        ('diningoffice', 'kitchen'),
        ('livingroom', 'longhall'),
        ('longhall', 'momsbathroom'),
        ('longhall', 'momsroom');
INSERT INTO portal_existence (destination, origin) SELECT origin, destination FROM portal_existence;
INSERT INTO portal_existence (origin, destination) VALUES
('guestroom', 'outside'),
         ('diningoffice', 'outside'),
         ('momsroom', 'outside'),
         ('myroom', 'outside');
