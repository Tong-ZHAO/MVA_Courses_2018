project(feature)

cmake_minimum_required(VERSION 2.8.11)

set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CGAL_DIR /home/pkq/cgal/cgal-build-release)
set(BOOST_DIR /usr/local/lib)

# Add specific Find.cmake modules 
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules) 

# Use C++11 for this directory and its sub-directories.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -fvisibility=hidden" )

find_package(CGAL QUIET COMPONENTS)
include( ${CGAL_USE_FILE} )

find_package( Boost REQUIRED COMPONENTS serialization iostreams)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
if( EIGEN3_FOUND)
  include(${EIGEN3_USE_FILE})
endif()

find_package( TBB )
if( TBB_FOUND )
  include(${TBB_USE_FILE})
  list(APPEND CGAL_3RD_PARTY_LIBRARIES ${TBB_LIBRARIES})
endif()

find_package(OpenCV QUIET)

include( CGAL_CreateSingleSourceCGALProgram )
set(needed_cxx_features cxx_rvalue_references cxx_variadic_templates)
create_single_source_cgal_program( "calc_feature.cpp" CXX_FEATURES ${needed_cxx_features})