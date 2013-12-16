--This file is part of LiSE, a development toolkit for life simulators.
--Copyright (C) Zachary Spector 2013
UPDATE img SET stacking_height=8 WHERE name IN ('blind', 'brownstone', 'orange');
UPDATE img SET stacking_height=12 WHERE name='block';
UPDATE img SET stacking_height=11 WHERE name='brutalist';
UPDATE img SET stacking_height=6 WHERE name IN ('crossroad', 'enterprise', 'monolith', 'olivine', 'soviet', 'lobby');
UPDATE img SET stacking_height=5 WHERE name IN ('sidewalk', 'street-ne-sw', 'street-nw-se');
UPDATE img SET stacking_height=4, off_y=-2 WHERE name='spacer';
UPDATE img SET off_y=2 WHERE name IN ('block', 'brutalist');
UPDATE img SET off_y=1 WHERE name='enterprise';
UPDATE img SET off_y=-1 WHERE name IN ('lobby', 'street-ne-sw', 'street-nw-se');
UPDATE img SET off_x=1 WHERE name='brownstone';
