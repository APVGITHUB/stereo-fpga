cmake_minimum_required(VERSION 3.18)

project(Open3DCMakeFindPackage LANGUAGES C CXX)

# Find installed Open3D, which exports Open3D::Open3D
find_package(Open3D REQUIRED)

add_executable(Draw)
target_sources(Draw PRIVATE Draw.cpp)
target_link_libraries(Draw PRIVATE Open3D::Open3D)


