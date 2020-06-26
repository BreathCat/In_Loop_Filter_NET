cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)
set(Torch_DIR "/home/li/Downloads/libtorch/share/cmake/Torch")

find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH
        /home/li/Downloads/libtorch
        /home/li/anaconda3/envs/py3.6/share/OpenCV
	)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}" "${OpenCV_LIBS}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
