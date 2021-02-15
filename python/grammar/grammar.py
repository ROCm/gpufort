#!/usr/bin/env python3
import os

exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar_options.py.in")).read())
if CASELESS:
    CASELESS_LITERAL = CaselessLiteral
else:
    CASELESS_LITERAL = Literal
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar_f03.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar_cuf.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar_acc.py.in")).read())
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar_epilog.py.in")).read())
