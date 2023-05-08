import pathlib
import os
parent_dir = pathlib.Path(__file__).parent.resolve()

intrinsics = set()

def load_intrinsics(file_name):
    global intrinsics
    with open(os.path.join(parent_dir,file_name),"r") as infile:
        intrinsics.update(infile.read().splitlines())

load_intrinsics("gfortran_intrinsics.inp")

incomplete_pgi_intrinsics = ['abs', 'aimag', 'aint', 'anint', 'ceiling', 'cmplx', 'conjg', 'dim', 'floor', 'int', 'logical', 'max', 'min', 'max0', 'min0', 'mod', 'modulo', 'nint', 'real', 'sign', 'acos', 'acosh', 'asin', 'asinh', 'atan', 'atanh', 'atan2', 'bessel_j0', 'bessel_j1', 'bessel_jn', 'bessel_y0', 'bessel_y1', 'bessel_yn', 'cos', 'cosh', 'erf', 'erfc', 'exp', 'gamma', 'hypot', 'log', 'log10', 'log_gamma', 'sin', 'sinh', 'sqrt', 'tan', 'tanh', 'bit_size', 'digits', 'epsilon', 'huge', 'maxexponent', 'minexponent', 'precision', 'radix', 'range', 'selected_int_kind', 'selected_real_kind', 'tiny', 'btest', 'iand', 'ibclr', 'ibits', 'ibset', 'ieor', 'ior', 'ishft', 'ishftc', 'leadz', 'mvbits', 'not', 'popcnt', 'poppar', 'all', 'any', 'count', 'maxloc', 'maxval', 'minloc', 'minval', 'product', 'sum', 'ubound', 'lbound', 'float', 'nint', 'sin1', 'cos1', 'tan1', 'amin', 'amax', 'amin1', 'amax1', 'aabs', 'asqrt', 'asin', 'asin1', 'acos', 'acos1', 'atan', 'atan1', 'alog', 'alog10', 'dexp', 'dlog10']

intrinsics.update(incomplete_pgi_intrinsics)
