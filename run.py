import argparse

parser = argparse.ArgumentParser()

# parses blue and red teams
parser.add_argument("blue", default="")
parser.add_argument("red", default="")

parser.add_argument("mode", default="predict", choices=("predict", "best"))
parser.add_argument("available", default="")


args = parser.parse_args()

import lol_prediction

# TODO: run prediction based on arguments...