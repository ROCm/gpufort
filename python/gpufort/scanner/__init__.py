
from .parser import *
from . import tree

#from . import opts # does not work; does create new opts module that differs from the local version
from .parser import opts # imports local opts module from a local module
