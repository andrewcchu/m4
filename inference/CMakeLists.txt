cmake_minimum_required(VERSION 3.16)
project(m4_native LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CMAKE_PREFIX_PATH is the active Python Torch's torch/share/cmake directory.
# Link its packaged LibTorch directly: TorchConfig asks for a CUDA toolkit even
# when the wheel already carries the runtime libraries needed for this target.
if(NOT CMAKE_PREFIX_PATH)
    message(FATAL_ERROR "Pass -DCMAKE_PREFIX_PATH=$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')")
endif()
get_filename_component(TORCH_ROOT "${CMAKE_PREFIX_PATH}/../.." ABSOLUTE)
if(NOT EXISTS "${TORCH_ROOT}/include/torch/csrc/api/include/torch/torch.h" OR
   NOT EXISTS "${TORCH_ROOT}/lib/libtorch.so")
    message(FATAL_ERROR "CMAKE_PREFIX_PATH does not point to an installed Python Torch package")
endif()
find_package(Python3 COMPONENTS Interpreter REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
    OUTPUT_VARIABLE TORCH_CXX11_ABI OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE TORCH_QUERY_RESULT)
if(NOT TORCH_QUERY_RESULT EQUAL 0 OR NOT TORCH_CXX11_ABI MATCHES "^[01]$")
    message(FATAL_ERROR "Python3_EXECUTABLE must be the Python used by this Torch installation")
endif()
add_executable(no_flowsim main_m4_noflowsim.cpp)
target_include_directories(no_flowsim PRIVATE
    "${TORCH_ROOT}/include"
    "${TORCH_ROOT}/include/torch/csrc/api/include")
target_compile_definitions(no_flowsim PRIVATE _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX11_ABI})
target_compile_options(no_flowsim PRIVATE -O3)
target_link_libraries(no_flowsim PRIVATE
    "${TORCH_ROOT}/lib/libtorch.so"
    "${TORCH_ROOT}/lib/libtorch_cpu.so"
    "${TORCH_ROOT}/lib/libtorch_cuda.so"
    "${TORCH_ROOT}/lib/libc10.so")
set_target_properties(no_flowsim PROPERTIES BUILD_RPATH "${TORCH_ROOT}/lib")
