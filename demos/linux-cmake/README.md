# About the demo

This demo runs in command line. It presents how to use the library with linear and nonlinear models. The user can also provide some parameters in command line.

# Dependencies

## Matplot++

Matplot++ is a library for ploting std::vector variables. It is very useful when testing the filters. 
The library can be found here: https://alandefreitas.github.io/matplotplusplus/

Download the sources in a foder ant install it with support to Cmake file command `find_package`:

```bash
cmake --preset=system
cmake --build --preset=system
sudo cmake --install build/system
```

You can now use it from CMake with `find_package`:

```cmake
find_package(Matplot++ REQUIRED)
target_link_libraries(<your target> Matplot++::matplot)

# Compiling

From this foder, create the building dir:

```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles" (may need sudo or permissions)

Now, and in every change of project files, compile again:

```bash
make

The result of the compilation can be found at folder \p build\bin.

# Running

Run the following to se a list of options:

```bash
./bin/demo