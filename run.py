import argparse
from pprint import pprint
from distutils.util import strtobool
from datetime import datetime

import gym
import numpy as np

import agents

def bool_type(x):
  return bool(strtobool(x))

def parse_args():
  parser = argparse.ArgumentParser(
    description=("An information-theoretic approach to"
                 " curiosity-driven reinforcement learning"))


  parser.add_argument("-v", "--verbose",
                      type=bool_type,
                      default=False,
                      help="Verbose")

  parser.add_argument("--num_runs",
                      type=int,
                      default=1,
                      help="Number runs for the experiment")

  parser.add_argument("--num_episodes",
                      type=int,
                      default=1000,
                      help="Number episodes per run")

  DEFAULT_AGENT_CLS = "DefaultAgent"
  parser.add_argument("--agent_class",
                      type=str,
                      default=DEFAULT_AGENT_CLS,
                      help=("Name of the class (defined in agents.py) to be"
                            " used as an agent. Defaults to '{0}', i.e."
                            "agents.{0}."
                            "".format(DEFAULT_AGENT_CLS)))

  parser.add_argument("--results-file", default=None, type=str,
                      help="File to write results to")

  args = vars(parser.parse_args())

  return args

def main(args):
  pass

if __name__ == "__main__":
  args = parse_args()
  main(args)
