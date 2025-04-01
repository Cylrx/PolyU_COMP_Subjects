# SmartPark: A Smart Parking Management System for PolyU Using OS-Level Process Coordination

Yuqi Wang, Xikun Yang, Siyuan Liu, Yixiao Jin

> **Important for Everyone!**
> 
> The entry point of this program is in `src/SPMS.c`.

> **Implementation Note**
> 
> We implemented various data structures here, including a Dynamic Array (refer to `src/vector.h` and `src/vector.c`) and a Lazy Propagation Segment Tree (refer to `src/segtree.h` and `src/segtree.c`). As for schedulers, we implemented three scheduling algorithms: **First-Come-First-Serve (FCFS), Priority Scheduling (PRIO), and Optimal Scheduling (OPTI)**. **The OPTI scheduler is implemented using Longest-Job-First and Simulated Annealing (SA), and is therefore also referred to as the LJF-SA Scheduler.**

## Build Guide

### Build with `gcc`
Please create a project folder first. Then move the `src/` folder into the project folder. Finally, build the whole project folder. 

Use the `cd` command to change the working directory to the project folder, then run this `gcc` command to build the project:
```shell
gcc -I./src src/*.c -o SPMS -lm -Wall -Wextra -Wshadow -Wconversion  -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -O2
```

Or alternatively,

```shell
gcc -I./src src/SPMS.c src/input.c src/opti.c src/output.c src/rng.c src/scheduler.c src/segtree.c src/state.c src/utils.c src/vector.c -o SPMS -lm Wall -Wextra -Wshadow -Wconversion  -D_GLIBCXX_DEBUG -D_GLIBCXX_DEBUG_PEDANTIC -O2
```

### Build with CMake

Since the CMake for this project was created using JetBrains CLion, we recommend that you create a new project in CLion, move the `src/` folder to the project directory, then add the following code to the `CMakeLists.txt` file. Afterward, use "Reload CMake Project" to apply the changes. Finally, you can run the entire project.

```CMake
cmake_minimum_required(VERSION 3.30)
project(COMP2432_GroupProject_SPMS C)

set(CMAKE_C_STANDARD 11)

file(GLOB SOURCES "src/*.c")

add_compile_definitions(DEBUG_MODE)

add_executable(COMP2432_GroupProject_SPMS ${SOURCES})

set(CMAKE_BUILD_TYPE Debug)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(
            -g
            -D_GLIBCXX_DEBUG
            -D_GLIBCXX_DEBUG_PEDANTIC
    )
else()
    add_compile_options(
            -O2
    )
endif()
```
