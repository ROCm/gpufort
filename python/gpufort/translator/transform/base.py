from .. import tree

class Transformer:
    
    def __init__(self):
        self.self._counters = {}

    def unique_c_identifier(self, label):
        """Returns a unique label for a loop variable that describes
        a loop entity. The result is prefixed with "_" to
        prevent collisions with Fortran variables.
        """
        if label not in self._counters:
            self._counters[label] = 0
        num = self._counters[label]
        self._counters[label] += 1
        return tree.TTCIdentifier("_"+label+str(num))