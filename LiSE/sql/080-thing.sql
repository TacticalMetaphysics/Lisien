-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO thing (character, name, host) VALUES
       ('household', 'zack_body', 'Physical'),
       ('household', 'gail_body', 'Physical');
INSERT INTO thing_loc (character, name, location) VALUES
       ('household', 'zack_body', 'zack_room'),
       ('household', 'gail_body', 'gail_room');
