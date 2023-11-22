import os, sys


def fix_relative_imports():
    dir2 = os.path.abspath("")
    dir1 = os.path.dirname(dir2)
    if not dir1 in sys.path:
        sys.path.append(dir1)
