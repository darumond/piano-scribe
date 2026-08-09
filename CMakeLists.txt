cmake_minimum_required(VERSION 3.20)
project(piano_transcriber_native VERSION 0.1.0 LANGUAGES CXX)

option(PIANO_TRANSCRIBER_BUILD_TESTS "Build native unit tests" ON)

add_library(piano_transcriber_dsp STATIC
    cpp/src/audio.cpp
    cpp/src/dsp.cpp
)
target_include_directories(piano_transcriber_dsp PUBLIC cpp/include)
target_compile_features(piano_transcriber_dsp PUBLIC cxx_std_17)
set_target_properties(piano_transcriber_dsp PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(MSVC)
    target_compile_options(piano_transcriber_dsp PRIVATE /W4 /permissive-)
else()
    target_compile_options(piano_transcriber_dsp PRIVATE -Wall -Wextra -Wpedantic)
endif()
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)
pybind11_add_module(_native cpp/src/bindings.cpp)
target_link_libraries(_native PRIVATE piano_transcriber_dsp)
target_compile_features(_native PRIVATE cxx_std_17)
install(TARGETS _native LIBRARY DESTINATION piano_transcriber RUNTIME DESTINATION piano_transcriber)

if(PIANO_TRANSCRIBER_BUILD_TESTS)
    enable_testing()
    add_executable(piano_transcriber_cpp_tests cpp/tests/test_dsp.cpp)
    target_link_libraries(piano_transcriber_cpp_tests PRIVATE piano_transcriber_dsp)
    target_compile_features(piano_transcriber_cpp_tests PRIVATE cxx_std_17)
    add_test(NAME piano_transcriber_cpp_tests COMMAND piano_transcriber_cpp_tests)
endif()
