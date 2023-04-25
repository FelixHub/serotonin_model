import os

ROOT_DIR = os.path.dirname(os.path.abspath(os.path.join(__file__, os.pardir)))
DATA_PATH = ROOT_DIR + "data/"

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)
