# This file is part of allegedb, an object-relational mapper for versioned graphs.
# Copyright (C) Zachary Spector. public@zacharyspector.com
#
# TODO: cancel changes that would put something back to where it was at the start
# This will complicate the update_window functions though, and I don't think it'll
# improve much apart from a bit of efficiency in that the deltas are smaller
# sometimes.
def update_window(turn_from, tick_from, turn_to, tick_to, updfun, branchd):
    """Iterate over a window of time in ``branchd`` and call ``updfun`` on the values"""
    if turn_from in branchd:
        # Not including the exact tick you started from because deltas are *changes*
        for past_state in branchd[turn_from][tick_from+1:]:
            updfun(*past_state)
    for midturn in range(turn_from+1, turn_to):
        if midturn in branchd:
            for past_state in branchd[midturn][:]:
                updfun(*past_state)
    if turn_to in branchd:
        for past_state in branchd[turn_to][:tick_to]:
            updfun(*past_state)


def update_backward_window(turn_from, tick_from, turn_to, tick_to, updfun, branchd):
    """Iterate backward over a window of time in ``branchd`` and call ``updfun`` on the values"""
    if turn_from in branchd:
        for future_state in reversed(branchd[turn_from][:tick_from]):
            updfun(*future_state)
    for midturn in range(turn_from-1, turn_to, -1):
        if midturn in branchd:
            for future_state in reversed(branchd[midturn][:]):
                updfun(*future_state)
    if turn_to in branchd:
        for future_state in reversed(branchd[turn_to][tick_to+1:]):
            updfun(*future_state)