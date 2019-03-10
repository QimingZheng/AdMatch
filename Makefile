BUILD = .
DIR_SRC = ./src
DIR_INCLUDE = ./include
DIR_TOOLS = ./tools
DIR_LIB = ${BUILD}/lib
DIR_OBJ = ./src
DIR_BIN = ${BUILD}/bin

CPP_SRC = $(wildcard ${DIR_SRC}/*.cpp)
CUDA_SRC = $(wildcard ${DIR_SRC}/*.cu)
CPP_OBJ = $(patsubst %.cpp, ${DIR_OBJ}/%.o, $(notdir ${CPP_SRC}))
CUDA_OBJ = $(patsubst %.cu, ${DIR_OBJ}/%.o, $(notdir ${CUDA_SRC}))

TARGET = ${DIR_LIB}/libita.so

CC = nvcc
CUDA_INCLUDE = -I./ -I/mnt/nvmebeegfs/zqm/summer/workspace/cuda-8.0/include # -L/mnt/nvmebeegfs/zqm/summer/workspace/boost_1_68_0/build/lib
CCFLAGS += $(CUDA_INCLUDE) -arch=compute_37 -code=sm_37 -std=c++11 -Xcompiler -fopenmp --compiler-options "-fPIC"
DYN_LIB = -lboost_regex

.PHONY: all clean title tool

all: title $(TARGET) tool

$(TARGET) : ${CPP_OBJ} ${CUDA_OBJ}
	$(CC) $(CCFLAGS) -shared -o $(TARGET) ${CPP_OBJ} ${CUDA_OBJ} $(DYN_LIB)

${CPP_OBJ}: %.o : %.cpp
	$(CC) $(CCFLAGS) -dc -c -o $@ $<

${CUDA_OBJ}: %.o : %.cu
	$(CC) $(CCFLAGS) -dc -c -o $@ $<

title:
	mkdir -p $(DIR_LIB) $(DIR_BIN)

tool: $(TARGET)
	$(CC) -I$(DIR_INCLUDE) $(CUDA_INCLUDE) -L$(DIR_LIB) -arch=compute_37 -code=sm_37 -std=c++11 -Xcompiler -fopenmp -o $(DIR_BIN)/sequence tools/sequence.cpp -lita -lboost_regex
	$(CC) -I$(DIR_INCLUDE) $(CUDA_INCLUDE) -L$(DIR_LIB) -arch=compute_37 -code=sm_37 -std=c++11 -Xcompiler -fopenmp -o $(DIR_BIN)/batch tools/batch.cpp -lita -lboost_program_options

clean:
	rm -f src/*.o $(TARGET) $(DIR_BIN)/sequence $(DIR_BIN)/batch
