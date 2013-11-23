-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO pawn_img (thing, img, layer) VALUES
       ('me', 'human_m', 0),
       ('me', 'edison', 1),
       ('mom', 'human_f', 0),
       ('mom', 'dress_green', 1);
INSERT INTO pawn_interactive (thing) SELECT DISTINCT thing FROM pawn_img;
