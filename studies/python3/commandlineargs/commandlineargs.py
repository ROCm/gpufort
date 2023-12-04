import sys

options = sys.argv[1:]
print(options)

prefix="--gpufort"
targets=["host","target"]
flags=["compiler","cflags","ldflags"]

compiler_options    = ["-".join([prefix,x,y]) for x in targets for y in flags]
result = {}

print(compiler_options)

current_key     = None
for i, opt in enumerate(list(options)):
    if opt in compiler_options:
        current_key = opt
        result[current_key] = []
    else:
         if current_key != None:
             result[current_key].append(opt)

print(result)
