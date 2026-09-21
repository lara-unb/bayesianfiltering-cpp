# cd SomeUnixFolder
# mkdir build
# cd build
# cmake .. -G "Unix Makefiles" (may need sudo or permissions)
# make (may need sudo or permissions)

# --------------------------------------------------------------
# Compiler Flags
# --------------------------------------------------------------

# specify the C++ standard

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

# bin folder
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# flags
# UNIX Multithread support flags
# All libs as static
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS}  -Wall -Werror=format -Wno-unused-function -std=c++17 -DNDEBUG -O0 -g ")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS}  -Wall -Werror=format -Wno-unused-function  -std=c++17 -DNDEBUG -O2 -s ")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}  -Wall -Werror=format -Wno-unused-function -std=c++17 -DNDEBUG -Ofast -s ")

# sources
set(BAYESIANFILTERING_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../../)

add_executable (demo 
    "main.cpp" )

install(TARGETS demo 
    DESTINATION ${CMAKE_INSTALL_PREFIX}
    )

# includes
target_include_directories(demo PUBLIC ${BAYESIANFILTERING_INCLUDE})

# Locate the Matplot++ package
find_package(Matplot++ REQUIRED)

# Locate the Python packages
#find_package(Python3 COMPONENTS Development)
#find_package(NumPy REQUIRED)
#find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

target_include_directories(demo PRIVATE ${Python3_INCLUDE_DIRS} ${NumPy_INCLUDE_DIRS})


# Locate the Eigen3 package
find_package(Eigen3 REQUIRED NO_MODULE)

target_link_libraries(demo PRIVATE Eigen3::Eigen Matplot++::matplot ${Python3_LIBRARIES})

#target_link_libraries(demo PRIVATE Eigen3::Eigen ${Python3_LIBRARIES})
