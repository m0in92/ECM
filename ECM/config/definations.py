import os


ROOT_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), '..'))
PROJ_DIR = os.path.relpath(os.path.join(os.path.dirname(__file__), '..\..'))
DATA_DIR = os.path.join(PROJ_DIR, 'data')
PARAMETER_SET_DIR = os.path.join(PROJ_DIR, 'parameter_sets')