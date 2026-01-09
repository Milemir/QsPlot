cmake_minimum_required(VERSION 3.15)
project(QsPlot)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Force Python Include if provided (Fix for missing headers in nanobind-static)
# Force Python Target if provided (Fix for missing headers/libs in nanobind-static)
if(DEFINED MANUAL_PYTHON_INCLUDE)
    message(STATUS "Manually defining Python::Module target")
    
    # We assume libs is parallel to include in standard python layout
    cmake_path(GET MANUAL_PYTHON_INCLUDE PARENT_PATH PYTHON_ROOT)
    set(PYTHON_LIB_PATH "${PYTHON_ROOT}/libs/python313.lib")
    message(STATUS "DEBUG: Manual Python Lib Path: ${PYTHON_LIB_PATH}")
    
    if(NOT TARGET Python::Module)
        # GLOBAL is essential so nanobind (in infra/dep/) sees this target!
        add_library(Python::Module SHARED IMPORTED GLOBAL)
        set_target_properties(Python::Module PROPERTIES
            IMPORTED_LOCATION "${PYTHON_LIB_PATH}"
            IMPORTED_IMPLIB "${PYTHON_LIB_PATH}"
            INTERFACE_INCLUDE_DIRECTORIES "${MANUAL_PYTHON_INCLUDE}"
        )
    endif()
    
    # Also add global include to be safe
    include_directories("${MANUAL_PYTHON_INCLUDE}")
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# In a real scenario, you might use FetchContent or git submodules.
# For this structure, we assume they are inside infra/dep/

# Check for Nanobind
# 1. Try finding it via CMake (e.g. from pip install nanobind)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG QUIET)

# 2. If not found, try the submodule
if(NOT nanobind_FOUND)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/nanobind/CMakeLists.txt")
        add_subdirectory(infra/dep/nanobind)
    else()
        message(FATAL_ERROR "Nanobind not found! (Neither via find_package nor infra/dep/nanobind)")
    endif()
endif()

# Check for GLFW
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/glfw/CMakeLists.txt")
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    add_subdirectory(infra/dep/glfw)
else()
    message(WARNING "GLFW not found in infra/dep/glfw. Please clone it.")
endif()

# Eigen (Header only)
set(EIGEN_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/eigen")
if(NOT EXISTS "${EIGEN_INCLUDE_DIR}/Eigen/Dense")
    message(WARNING "Eigen not found in infra/dep/eigen. Please clone it.")
endif()

# Glad (OpenGL Loader)
# We assume glad.c and glad.h are present in infra/dep/glad
add_library(glad "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/glad/src/glad.c")
target_include_directories(glad PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/glad/include")

# Check for ImGui
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/imgui/imgui.cpp")
    set(IMGUI_DIR "${CMAKE_CURRENT_SOURCE_DIR}/infra/dep/imgui")
    set(IMGUI_SOURCES
        "${IMGUI_DIR}/imgui.cpp"
        "${IMGUI_DIR}/imgui_demo.cpp"
        "${IMGUI_DIR}/imgui_draw.cpp"
        "${IMGUI_DIR}/imgui_tables.cpp"
        "${IMGUI_DIR}/imgui_widgets.cpp"
        "${IMGUI_DIR}/backends/imgui_impl_glfw.cpp"
        "${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp"
    )
    # Define IMGUI_IMPL_OPENGL_LOADER_GLAD to tell backend to use GLAD
    add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)
else()
    message(WARNING "ImGui not found in infra/dep/imgui.")
endif()

# -----------------------------------------------------------------------------
# Main Module
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Main Module
# -----------------------------------------------------------------------------
if(COMMAND nanobind_add_module)
    # Note: Target name must match the import name in Python
    # Scikit-build expects the install rule to place it correctly.
    nanobind_add_module(qsplot_engine 
        src/qsplot/bindings/bind_main.cpp 
        src/qsplot/core/DataProcessor.cpp 
        src/qsplot/graphics/Renderer.cpp
        src/qsplot/graphics/Renderer_Picking.cpp
        src/qsplot/graphics/Camera.cpp
        ${IMGUI_SOURCES}
    )

    target_include_directories(qsplot_engine PRIVATE 
        src/qsplot 
        ${EIGEN_INCLUDE_DIR}
        ${IMGUI_DIR}
        ${IMGUI_DIR}/backends
    )

    target_link_libraries(qsplot_engine PRIVATE 
        glad 
        glfw 
    )
    
    if(MSVC)
        target_compile_options(qsplot_engine PRIVATE /W4)
    else()
        target_compile_options(qsplot_engine PRIVATE -Wall -Wextra)
    endif()

    # Install directive for scikit-build
    # We install to '.' because scikit-build sets CMAKE_INSTALL_PREFIX to the package dir defined in setup.py
    install(TARGETS qsplot_engine LIBRARY DESTINATION .)

else()
    message(FATAL_ERROR "Nanobind target is missing. Cannot build module.")
endif()
