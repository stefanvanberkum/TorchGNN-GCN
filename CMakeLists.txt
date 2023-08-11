cmake_minimum_required(VERSION 3.10)

project(TorchGNN)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch REQUIRED)
find_package(TorchScatter REQUIRED)
find_package(TorchSparse REQUIRED)
find_package(BLAS REQUIRED)

add_executable(TorchGNN main.cxx)

add_subdirectory(Model)
target_link_libraries(TorchGNN PUBLIC Model)
target_link_libraries(TorchGNN PUBLIC ${BLAS_LIBRARIES})
target_link_libraries(TorchGNN PUBLIC ${TORCH_LIBRARIES})
target_link_libraries(TorchGNN PUBLIC TorchScatter::TorchScatter)
target_link_libraries(TorchGNN PUBLIC TorchSparse::TorchSparse)
