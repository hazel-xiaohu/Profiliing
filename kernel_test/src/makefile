include ../../make.inc
ifdef gpu
all: pangulu_cuda.o 

pangulu_cuda.o:pangulu_cuda.cu
	$(NVCC) $(NVCCFLAGS) -Xcompiler -fPIC  -c $< -o $@ 

endif
clean:
	rm -f pangulu_cuda.o