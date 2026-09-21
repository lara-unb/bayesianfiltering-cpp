# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#


cmake_minimum_required (VERSION 3.15)

if (UNIX)
    # Define project name.
    project ("demo" VERSION 1.0)
    # Include sub-projects.
    add_subdirectory ("src")
endif()


