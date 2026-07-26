include ../../make.inc
Kernel_INC = -I ../src
SRC = $(wildcard TEST_SRC*.cpp)
OBJ = $(patsubst %.cpp,%.o,$(SRC))

ifdef gpu
PANGULU_CUDA=../src/pangulu_cuda.o
endif

ifdef gpu
all:ssssm getrf gessm tstrf

ssssm : pangulu_ssssm_test.o ${PANGULU_CUDA} 
	$(CXX)  $^ $(CPPFLAGS) $(CUDA_INC) $(CUDA_LIB) $(Kernel_INC) -o $@

getrf : pangulu_getrf_test.o ${PANGULU_CUDA} 
	$(CXX)  $^ $(CPPFLAGS) $(CUDA_INC) $(CUDA_LIB) $(Kernel_INC) -o $@

gessm : pangulu_gessm_test.o ${PANGULU_CUDA} 
	$(CXX)  $^ $(CPPFLAGS) $(CUDA_INC) $(CUDA_LIB) $(Kernel_INC) -o $@

tstrf : pangulu_tstrf_test.o ${PANGULU_CUDA} 
	$(CXX)  $^ $(CPPFLAGS) $(CUDA_INC) $(CUDA_LIB) $(Kernel_INC) -o $@

%.o : %.cpp
	$(CXX)  $(CPPFLAGS) $(Kernel_INC)  -c $< -o $@
else
all:ssssm getrf gessm tstrf
ssssm : pangulu_ssssm_test.o 
	$(CXX)  $^ $(CPPFLAGS) $(Kernel_INC) -o $@

getrf : pangulu_getrf_test.o
	$(CXX)  $^ $(CPPFLAGS) $(Kernel_INC) -o $@

gessm : pangulu_gessm_test.o
	$(CXX)  $^ $(CPPFLAGS) $(Kernel_INC) -o $@

tstrf : pangulu_tstrf_test.o
	$(CXX)  $^ $(CPPFLAGS) $(Kernel_INC) -o $@

%.o : %.cpp
	$(CXX)  $(CPPFLAGS) $(Kernel_INC)  -c $< -o $@
endif

.PHONY:clean

clean:
	@rm -rf *.o ${CUDA_SRC}/pangulu_cuda.o
	-rm -rf ssssm getrf gessm tstrf
test:
	@echo $(SRC)
	@echo $(OBJ)
distclean: