import addtoplevelpath

import fort2x.derivedtypegen
import indexer.scoper as scoper

class HipDerivedTypeGenerator(fort2x.derivedtypegen.DerivedTypeGenerator):
    def __init__(self,
                 itypes,
                 used_modules=[{"name" : mod,"only" : []} for mod in ["hipfort","hipfort_check","gpufort_array"]]):
        fort2x.derivedtypegen.DerivedTypeGenerator.__init__(self,
                                                            itypes,
                                                            used_modules)
        self.synchronize_queue = "hipStreamSynchronize"
        self.error_check       = "hipCheck"
