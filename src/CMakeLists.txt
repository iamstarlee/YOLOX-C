cmake_minimum_required(VERSION 3.10)

set(LIBRARY_NAME ${PROJECT_NAME})

find_library(onnxruntime_LIBS NAMES onnxruntime PATHS /usr/local/lib)

file(GLOB SOURCE_FILES
  ${PROJECT_SOURCE_DIR}/src/ImageRecognitionOrtSessionHandlerBase.cpp
  ${PROJECT_SOURCE_DIR}/src/OrtSessionHandler.cpp
)

include_directories(/home/whoami/Documents/codes/YOLOX-C/onnxruntime/include/onnxruntime/ /home/whoami/Documents/codes/YOLOX-C/onnxruntime/onnxruntime/)
add_library(${LIBRARY_NAME}
  SHARED
     ${SOURCE_FILES}
)

target_include_directories(${LIBRARY_NAME}
  PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  PRIVATE
    ${onnxruntime_INCLUDE_DIRS}
)

target_compile_options(${LIBRARY_NAME}
  PRIVATE
     $<$<CONFIG:Debug>:-O0 -g -Wall -Werror>
     $<$<CONFIG:Release>:-O3>
)

list(APPEND PRIVATE_LIBS ${onnxruntime_LIBS})

target_link_libraries(${LIBRARY_NAME}
  PRIVATE
    ${PRIVATE_LIBS}
)
