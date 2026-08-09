cmake_minimum_required(VERSION 3.20)
project(piano_scribe_native VERSION 0.1.0 LANGUAGES CXX)

option(PIANO_SCRIBE_BUILD_TESTS "Build native unit tests" ON)

add_library(piano_scribe_dsp STATIC
    cpp/src/audio.cpp
    cpp/src/dsp.cpp
)
target_include_directories(piano_scribe_dsp PUBLIC cpp/include)
target_compile_features(piano_scribe_dsp PUBLIC cxx_std_17)
set_target_properties(piano_scribe_dsp PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(piano_scribe_dsp PRIVATE /W4 /permissive-)
else()
    target_compile_options(piano_scribe_dsp PRIVATE -Wall -Wextra -Wpedantic)
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native cpp/src/bindings.cpp)
target_link_libraries(_native PRIVATE piano_scribe_dsp)
target_compile_features(_native PRIVATE cxx_std_17)
install(TARGETS _native LIBRARY DESTINATION piano_transcriber RUNTIME DESTINATION piano_transcriber)

if(PIANO_SCRIBE_BUILD_TESTS)
    enable_testing()
    add_executable(piano_scribe_cpp_tests cpp/tests/test_dsp.cpp)
    target_link_libraries(piano_scribe_cpp_tests PRIVATE piano_scribe_dsp)
    target_compile_features(piano_scribe_cpp_tests PRIVATE cxx_std_17)
    add_test(NAME piano_scribe_cpp_tests COMMAND piano_scribe_cpp_tests)
endif()
