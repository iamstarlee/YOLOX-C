cmake_minimum_required(VERSION 3.10)

include(cmake_utility)

find_package(OpenCV REQUIRED)

list(APPEND PUBLIC_LIBS
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
)

list(APPEND PRIVATE_LIBS)

foreach(EXAMPLE ${EXAMPLES})
  __build_example(
    ${EXAMPLE}
    "${PUBLIC_LIBS}"
    "${PRIVATE_LIBS}"
  )
  target_include_directories(${EXAMPLE}
    PUBLIC
      ${OpenCV_INCLUDE_DIRS}
  )
endforeach(EXAMPLE)

# ---------------------------------------------------------

add_executable(yolox
  ${CMAKE_CURRENT_LIST_DIR}/YoloX.cpp
  ${CMAKE_CURRENT_LIST_DIR}/YoloX.hpp
  ${CMAKE_CURRENT_LIST_DIR}/YoloXApp.cpp
)

target_link_libraries(yolox
  PUBLIC
    ${PROJECT_NAME}
    ${OpenCV_LIBS}
)

target_include_directories(yolox
  PUBLIC
    ${OpenCV_INCLUDE_DIRS}
)