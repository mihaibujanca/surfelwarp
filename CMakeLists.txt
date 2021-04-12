cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(surfelwarp)

# Add cmake module path
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
if(${SLAMBENCH_AVAILABLE})
    message("Building SurfelWarp with PIC enabled")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler=-fPIC")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# Language options
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(VISUALIZATION_LIBRARY "cilantro")

# Cuda
find_package(CUDA 9 REQUIRED)
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_ARCH "72" CACHE STRING "Architecture(s) for which to generate CUDA PTX code")
if(CUDA_VERSION_MAJOR LESS 10)
    set_property(CACHE CUDA_ARCH PROPERTY STRINGS "50 through 61" "50" "60" "61")
else()
    set_property(CACHE CUDA_ARCH PROPERTY STRINGS "50 through 61" "50" "60" "61" "70" "72" "75")
endif()
if(CUDA_ARCH STREQUAL "50 through 61")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_50,code=sm_50")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_60,code=sm_60")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode=arch=compute_61,code=sm_61")
else()
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -arch=sm_${CUDA_ARCH};-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}")
endif()

# GLFW package
find_package(glfw3 REQUIRED)


# The library and compiler flags on Windows
if(MSVC)
    option( MSVC_USE_STATIC_CRT  "Use static C Runtime with MSVC, /MT instead of /MD" ON)

    # Make sure there are no erroneous C Runtime flags
    list(APPEND FLAG_VARS
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CMAKE_C_FLAGS CMAKE_C_FLAGS_DEBUG CMAKE_C_FLAGS_RELEASE CMAKE_C_FLAGS_MINSIZEREL CMAKE_C_FLAGS_RELWITHDEBINFO
    )
    if(MSVC_USE_STATIC_CRT)
        foreach(FLAG_VAR ${FLAG_VARS})
            string(REGEX REPLACE "/MD" "/MT" NEW_FLAGS "${${FLAG_VAR}}")
            set(${FLAG_VAR} "${NEW_FLAGS}" CACHE STRING "" FORCE)
        endforeach()
    else()
        foreach(FLAG_VAR ${FLAG_VARS})
            string(REGEX REPLACE "/MT" "/MD" NEW_FLAGS "${${FLAG_VAR}}")
            set(${FLAG_VAR} "${NEW_FLAGS}" CACHE STRING "" FORCE)
        endforeach()
    endif()

	# Provisions for typical Boost compiled on Windows
    # Most commonly, Boost libraries are compiled statically on windows (change as necesssary)
    set(Boost_USE_STATIC_LIBS TRUE)
	if(MSVC_USE_STATIC_CRT)
		set(Boost_USE_STATIC_RUNTIME ON)
	else()
		set(Boost_USE_STATIC_RUNTIME OFF)
	endif()
	set(Boost_USE_MULTITHREADED ON)
	set(Boost_USE_DEBUG_RUNTIME OFF)
	set(Boost_USE_DEBUG_PYTHON OFF)
endif(MSVC)
# The library and compiler flags on Ubuntu
if(UNIX)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -std=c++14 -Xptxas -O3 -Xcompiler=-O3")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++17") # Ofast here
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
endif(UNIX)

# Append "d" to library versions with debugging symbols
set(CMAKE_DEBUG_POSTFIX d)

# Do not use the system Eigen
set(EIGEN_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/external/eigen3.4")
set(EIGEN_INCLUDE_DIR ${EIGEN_ROOT})
set(EIGEN3_INCLUDE_DIR ${EIGEN_ROOT})
include_directories(BEFORE ${EIGEN3_INCLUDE_DIR} ${EIGEN_INCLUDE_DIR})

add_definitions(-DCUB_USE_COOPERATIVE_GROUPS)
# The cub package
set(CUB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/cub-1.8.0")
include_directories(${CUB_DIR})

#set_property(CACHE VISUALIZATION_LIBRARY PROPERTY STRINGS "PCL" "cilantro")
if(VISUALIZATION_LIBRARY STREQUAL "PCL")
    find_package(PCL REQUIRED)
    include_directories(${PCL_INCLUDE_DIRS})
    add_definitions(-DWITH_PCL)
elseif(VISUALIZATION_LIBRARY STREQUAL "cilantro")
    find_package(cilantro REQUIRED)
    include_directories(${cilantro_INCLUDE_DIRS})
    add_definitions(-DWITH_CILANTRO)
endif()


# Check the CUDA version and resolve conflicts with pcl
if(${CUDA_VERSION_MAJOR} LESS 11)
    if(VISUALIZATION_LIBRARY STREQUAL "PCL")
        list(REMOVE_ITEM PCL_LIBRARIES "vtkproj4")
    endif()
    get_directory_property(dir_defs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} COMPILE_DEFINITIONS)
    set(vtk_flags)
    foreach(it ${dir_defs})
        if(it MATCHES "vtk*")
            list(APPEND vtk_flags ${it})
        endif()
    endforeach()

    foreach(d ${vtk_flags})
        remove_definitions(-D${d})
    endforeach()
endif()

# OpenCV lib: this version uses the customized build
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
include_directories(${OpenCV_INCLUDE_DIR})

# The boost library is contained in pcl
find_package(Boost REQUIRED COMPONENTS system filesystem)
include_directories(${Boost_INCLUDE_DIRS})

# The project directory shall also be included
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/external")

# The external library
add_subdirectory(external)

# The libraries
add_subdirectory(common)
add_subdirectory(math)
add_subdirectory(pcg_solver)
add_subdirectory(imgproc)
add_subdirectory(visualization)
add_subdirectory(core)


# The hashing library should not depend on PCL
add_subdirectory(hashing)

# Applications
add_subdirectory(apps)
