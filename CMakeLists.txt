cmake_minimum_required (VERSION 2.8)

project (admatch)

find_package(Boost COMPONENTS chrono system thread regex program_options REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${DISABLED_WARNINGS} -std=c++11 -fPIC")

INCLUDE_DIRECTORIES(
  ${PROJECT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}
  /usr/local/include
  /usr/include
)

set(DISABLED_WARNINGS "-Wno-ignored-attributes -Wno-deprecated-declarations")

file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
file(GLOB_RECURSE CUDA_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cu)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

string(APPEND CMAKE_CUDA_FLAGS " -rdc=true -Xcompiler -fopenmp --compiler-options -fPIC --expt-relaxed-constexpr -w")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_37,code=sm_37")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_52,code=sm_52")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_60,code=sm_60")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")
string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_70,code=sm_70")

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

add_library (admatch SHARED ${CPP_SOURCES} ${CUDA_SOURCES})
target_link_libraries(admatch cuda ${Boost_REGEX_LIBRARY} pthread)

add_subdirectory(tools)
