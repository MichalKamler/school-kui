# Some named tuples to be used throughout the package

import collections

#: Namedtuple to hold state position.
State = collections.namedtuple("State", ["x", "y"])

#: Namedtuple to hold path section from state A to state B. Expects C{state_from} and C{state_to} to be of type L{State}
PathSection = collections.namedtuple(
    "PathSection", ["state_from", "state_to", "cost", "action"]
)
