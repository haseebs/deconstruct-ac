
"""
Tile Coding Software version 3.0beta
by Rich Sutton
based on a program created by Steph Schaeffer and others
External documentation and recommendations on the use of this code is available in the 
reinforcement learning textbook by Sutton and Barto, and on the web.
These need to be understood before this code is.

This software is for Python 3 or more.

This is an implementation of grid-style tile codings, based originally on
the UNH CMAC code (see http://www.ece.unh.edu/robots/cmac.htm), but by now highly changed. 
Here we provide a function, "tiles", that maps floating and integer
variables to a list of tiles, and a second function "tiles-wrap" that does the same while
wrapping some floats to provided widths (the lower wrap value is always 0).

The float variables will be gridded at unit intervals, so generalization
will be by approximately 1 in each direction, and any scaling will have 
to be done externally before calling tiles.

Num-tilings should be a power of 2, e.g., 16. To make the offsetting work properly, it should
also be greater than or equal to four times the number of floats.

The first argument is either an index hash table of a given size (created by (make-iht size)), 
an integer "size" (range of the indices from 0), or nil (for testing, indicating that the tile 
coordinates are to be returned without being converted to indices).
"""
from enum import Enum

from numpy.core.fromnumeric import resize
from math import floor, log
from itertools import zip_longest
basehash = hash


class IHT:
  "Structure to handle collisions"

  def __init__(self, sizeval):
    self.size = sizeval
    self.overfullCount = 0
    self.dictionary = {}

  def __str__(self):
    "Prepares a string for printing whenever this object is printed"
    return "Collision table:" + \
           " size:" + str(self.size) + \
           " overfullCount:" + str(self.overfullCount) + \
           " dictionary:" + str(len(self.dictionary)) + " items"

  def count(self):
    return len(self.dictionary)

  def fullp(self):
    return len(self.dictionary) >= self.size

  def getindex(self, obj, readonly=False):
    d = self.dictionary
    if obj in d:
      return d[obj]
    elif readonly:
      return None
    size = self.size
    count = self.count()
    if count >= size:
      if self.overfullCount == 0:
        print('IHT full, starting to allow collisions')
      self.overfullCount += 1
      return basehash(obj) % self.size
    else:
      d[obj] = count
      return count


def hashcoords(coordinates, m, readonly=False):
  if type(m) == IHT:
    return m.getindex(tuple(coordinates), readonly)
  if type(m) == int:
    return basehash(tuple(coordinates)) % m
  if m == None:
    return coordinates


def tiles(ihtORsize, numtilings, floats, ints=[], readonly=False):
  """returns num-tilings tile indices corresponding to the floats and ints"""
  qfloats = [floor(f*numtilings) for f in floats]
  Tiles = []
  for tiling in range(numtilings):
    tilingX2 = tiling*2
    coords = [tiling]
    b = tiling
    for q in qfloats:
      coords.append((q + b) // numtilings)
      b += tilingX2
    coords.extend(ints)
    Tiles.append(hashcoords(coords, ihtORsize, readonly))
  return Tiles


def tileswrap(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=False):
  """returns num-tilings tile indices corresponding to the floats and ints, wrapping some floats"""
  qfloats = [floor(f*numtilings) for f in floats]
  Tiles = []
  for tiling in range(numtilings):
    tilingX2 = tiling*2
    coords = [tiling]
    b = tiling
    for q, width in zip_longest(qfloats, wrapwidths):
      c = (q + b % numtilings) // numtilings
      coords.append(c % width if width else c)
      b += tilingX2
    coords.extend(ints)
    Tiles.append(hashcoords(coords, ihtORsize, readonly))
  return Tiles

class StateFeatureType(Enum):
    INDEX = "index"
    VECTOR = "vector"


class TileCoder():
    def __init__(self, ndims, num_tilings, ranges, num_tiles):
        """Initialization

        Args:
            ndims (int): number of dimensions in the state
            num_tilings (int): number of tilings to use
            ranges (list of tuples): (min,max) ranges for each dimension
            num_tiles (list of ints): number of tiles to use for each dimension

        """
        assert ndims == len(ranges) == len(
            num_tiles), "Number of dimensions, length of ranges, and length of num tiles must match"

        self.feature_type = StateFeatureType.INDEX

        self.ndims = ndims
        self.num_tilings = num_tilings
        self.ranges = ranges
        self.num_tiles = num_tiles

        # Get total number of tiles
        tiles_per_tiling = 1
        for tiles in num_tiles:
            tiles_per_tiling *= tiles + 1

        self.total_tiles = tiles_per_tiling * num_tilings

        self.iht = IHT(self.total_tiles)
        pass

    def transform(self, state):
        """Transforms the state into the state feature

        Args:
            state (Numpy array): the state observation
        Returns:
            state_feature (Numpy array): A list of indices where the feature vector is activated.
        """

        if (self.ndims != len(state)):
            raise TypeError("Number of dimension on state doesn't match the dimension on tile coder")

        resized_state = []
        for ind, v in enumerate(state):
            new_v = (v - self.ranges[ind][0]) / ((self.ranges[ind]
                                                  [1] - self.ranges[ind][0])) * self.num_tiles[ind]
            resized_state.append(new_v)
        indices = tiles(self.iht, self.num_tilings, resized_state)
        return indices
