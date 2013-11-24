-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO pawn_img (thing, img, layer) VALUES
       ('me', 'hominid/base/human_m.bmp', 0),
       ('me', 'hominid/body/saruman.bmp', 1),
       ('mom', 'hominid/base/human_f.bmp', 0),
       ('mom', 'hominid/body/dress_green.bmp', 1);
INSERT INTO pawn_interactive (thing) SELECT DISTINCT thing FROM pawn_img;
