-- This file is part of LiSE, a framework for life simulation games.
-- Copyright (c) 2013 Zachary Spector,  zacharyspector@gmail.com
INSERT INTO pawn_img (thing, img) VALUES ('me', 'human_m'), ('mom', 'human_f');
INSERT INTO pawn_interactive (thing) SELECT thing FROM pawn_img;
