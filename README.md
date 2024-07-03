# YOLOX-C
## This is a repository for deploying [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) using [onnxruntime](https://github.com/microsoft/onnxruntime).

## How to install 
install onnxruntime from source with the following script
```bash
    # onnxruntime needs newer cmake version to build
    bash ./scripts/install_latest_cmake.bash

    # This place may be very slow, see the bash file for more detail.
    bash ./scripts/install_onnx_runtime.bash

    # dependencies to build apps
    bash ./scripts/install_apps_dependencies.bash
```

## How to build
**CPU**
```bash
make default

# build examples
make apps
```
**GPU with CUDA**
```bash
make gpu_default

make gpu_apps
```

## How to test
1. Download onnx model trained on COCO dataset from [HERE](https://github.com/Megvii-BaseDetection/YOLOX/tree/main/demo/ONNXRuntime), and test yolox_l, you can try with other yolox models also.

```bash
wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_l.onnx -O ./data/yolox_l.onnx
```

2. Test inference examples
```bash
./build/examples/yolox ./models/yolox_l.onnx ./data/matrix.jpg
```