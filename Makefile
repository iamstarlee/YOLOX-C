BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)
USE_GPU=ON

default:
	@mkdir -p build
	@cd build && cmake .. -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                              -DUSE_GPU=$(USE_GPU) \
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
							  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
                              $(CMAKE_ARGS)
	@cd build && make

gpu_default:
	@make default USE_GPU=ON

debug:
	@make default BUILD_TYPE=Debug

gpu_apps:
	@make apps USE_GPU=ON

clean:
	@rm -rf build*
