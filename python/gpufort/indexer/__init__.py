
from .indexer import *
from . import scope
from . import types
from . import props
from . import intrinsics

#from . import opts # might not work; might create new opts module that differs from the package-local opts module
from .indexer import opts # imports package-local opts module
