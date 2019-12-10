# A compatability wrapper for the pickle module. cPickle was removed
# in Python 3.
import sys

try:
  import cPickle as pickle
except ModuleNotFoundError:
  import pickle


def load(*args, **kwargs):
  if sys.version_info > (3, 0): # is using python3, uncomment it
    return pickle.load(*args, **kwargs, encoding='latin1')
  else:
    return pickle.load(*args, **kwargs)


def dump(*args, **kwargs):
  return pickle.dump(*args, **kwargs)
