# Utility functions
import os


def makedir(path: str):
    """Creates a new directory `path`
    """
    if not os.path.exists(path):
        os.mkdir(path)
    else:
        print(f'Path {path} already exists!')
 