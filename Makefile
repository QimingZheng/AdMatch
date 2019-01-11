DIR_SRC = ./src
DIR_INCLUDE = ./include
DIR_TOOLS = ./tools
DIR_LIB = ./lib
DIR_OBJ = ./src

CPP_SRC = $(wildcard ${DIR_SRC}/*.cpp)
CUDA_SRC = $(wildcard ${DIR_SRC}/*.cu)
CPP_OBJ = $(patsubst %.cpp, ${DIR_OBJ}/%.o, $(notdir ${CPP_SRC}))
CUDA_OBJ = $(patsubst %.cu, ${DIR_OBJ}/%.o, $(notdir ${CUDA_SRC}))

TARGET = ${DIR_LIB}/libita.so

CC = nvcc
CUDA_INCLUDE = -I/usr/local/cuda-9.1/include -L/home/qizhe/SoftWare/boost_1_68_0/build/lib -I./
CCFLAGS += $(CUDA_INCLUDE) -arch=compute_60 -code=sm_60 -std=c++11 -Xcompiler -fopenmp --compiler-options "-fPIC"
DYN_LIB = -lboost_regex

.PHONY: all clean title

all: title $(TARGET)

$(TARGET) : ${CPP_OBJ} ${CUDA_OBJ}
	$(CC) $(CCFLAGS) -shared -o $(TARGET) ${CPP_OBJ} ${CUDA_OBJ} $(DYN_LIB)

${CPP_OBJ}: %.o : %.cpp
	$(CC) $(CCFLAGS) -dc -c -o $@ $<

${CUDA_OBJ}: %.o : %.cu
	$(CC) $(CCFLAGS) -dc -c -o $@ $<

title:
	mkdir -p $(DIR_LIB)

clean:
	rm -f src/*.o $(TARGET) 
