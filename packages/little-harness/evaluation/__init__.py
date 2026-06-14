"""Tool-overlap evaluation suite.

Compares tools that can perform the same task (e.g. calculator vs bash for
arithmetic, read_file vs bash for reading) by running each candidate against
a set of prompts and recording pass/fail + wall time.
"""
