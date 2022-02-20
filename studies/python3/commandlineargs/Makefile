test:
	python3 commandlineargs.py --gpufort-host-compiler gfortran --gpufort-host-cflags -I/usr/include --gpufort-host-ldflags -L/usr/lib -lmpi --gpufort-target-compiler 'hipcc ' --gpufort-target-cflags "--offload-arch=gfx908 -fPIC"
