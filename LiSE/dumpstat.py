#!/usr/bin/python3
import sys
print(sys.argv[-1])
import pstats

stats = pstats.Stats()
stats.load_stats(sys.argv[-1] if not sys.argv[-1].endswith('dumpstat.py') else 'profile')
stats.sort_stats('cumtime').reverse_order().print_stats()
