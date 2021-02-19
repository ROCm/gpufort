#!/usr/bin/env python3
import addtoplevelpath
import sys
import test
import translator.translator
import grammar as translator

testdata = """
1 )
a_d )
psi_d )
2 * lda, ps_d, 1, 1.D0, psi_d, 1 )
spsi_d )
a_d )
1, spsi_d )
1, 1, spsi_d )
lda, ps_d, 1, 1, spsi_d )
lda, ps_d )
lda, ps_d, 1, 1, spsi_d, 1 )
2 * lda, ps_d, 1, 1, spsi_d, 1 )
2 * lda, ps_d, 1, 1.D0, spsi_d, 1 )
""".strip("\n").strip(" ").strip("\n").splitlines()

test.run(
   expression     = translator.callEnd,
   testdata       = testdata,
   tag            = "callEnd",
   raiseException = True
)
