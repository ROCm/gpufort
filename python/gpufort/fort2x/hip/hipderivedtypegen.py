

from .. import derivedtypegen


class HipDerivedTypeGenerator(derivedtypegen.DerivedTypeGenerator):

    def __init__(self,
                 itypes,
                 used_modules=[{
                     "name": mod,
                     "only": []
                 } for mod in ["hipfort", "hipfort_check", "gpufort_array"]]):
        derivedtypegen.DerivedTypeGenerator.__init__(self, itypes,
                                                     used_modules)
        self.synchronize_queue = "hipStreamSynchronize"
        self.error_check = "hipCheck"
