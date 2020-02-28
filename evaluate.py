from treccast import Treccast
import os


def evaluate(dir_path):
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            pass


def AP():
    pass


def nDCG():
    pass


evaluate('data/test_set/')
