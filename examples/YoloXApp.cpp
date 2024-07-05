/**
 * @file    YoloXApp.cpp
 *
 * @author  btran
 *
 */

#include <ort_utility/ort_utility.hpp>

#include "Utility.hpp"
#include "YoloX.hpp"
#include <time.h>

static const std::vector<std::string> MSCOCO_WITHOUT_BG_CLASSES(Ort::MSCOCO_CLASSES.begin() + 1,
                                                                Ort::MSCOCO_CLASSES.end());
static constexpr int64_t NUM_CLASSES = 80;
static const std::vector<std::array<int, 3>> COLOR_CHART = Ort::generateColorCharts(NUM_CLASSES);

static constexpr float CONFIDENCE_THRESHOLD = 0.1;
static const std::vector<cv::Scalar> COLORS = toCvScalarColors(COLOR_CHART);

namespace
{
cv::Mat processOneFrame(const Ort::YoloX& osh, const cv::Mat& inputImg, float* dst, const float confThresh);
}  // namespace

int main(int argc, char* argv[])
{
    
    if (argc != 3) {
        std::cerr << "Usage: [apps] [path/to/onnx/yolox] [path/to/image]" << std::endl;
        return EXIT_FAILURE;
    }

    const std::string ONNX_MODEL_PATH = argv[1];
    const std::string IMAGE_PATH = argv[2];
    const std::string OUTPUT_PATH = "results";

    // 实例化一个YoloX对象
    Ort::YoloX osh(
        NUM_CLASSES, ONNX_MODEL_PATH, 0,
        std::vector<std::vector<int64_t>>{{1, Ort::YoloX::IMG_CHANNEL, Ort::YoloX::IMG_H, Ort::YoloX::IMG_W}});

    osh.initClassNames(MSCOCO_WITHOUT_BG_CLASSES);

    std::vector<float> dst(Ort::YoloX::IMG_CHANNEL * Ort::YoloX::IMG_H * Ort::YoloX::IMG_W);

    // 遍历文件夹中的每个文件
    namespace fs = std::filesystem;
    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(IMAGE_PATH)) {
        if (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg") {
            
            
            cv::Mat image = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
             if (image.empty()) {
                std::cerr << "Warning: Could not read image " << entry.path().string() << std::endl;
                return EXIT_FAILURE;
            }
            else{
                auto result = processOneFrame(osh, image, dst.data(), CONFIDENCE_THRESHOLD);
                // 构造保存路径
                std::string savePath = OUTPUT_PATH + "/" + entry.path().filename().string();
                bool saved = cv::imwrite(savePath, result);
                if (saved) {
                    std::cout << "Saved image: " << savePath << std::endl;
                } else {
                    std::cerr << "Failed to save image: " << savePath << std::endl;
                }
            }
            
            
        }
    }
    return EXIT_SUCCESS;
}

namespace
{
cv::Mat processOneFrame(const Ort::YoloX& osh, const cv::Mat& inputImg, float* dst, const float confThresh)
{
    int origW = inputImg.cols, origH = inputImg.rows;
    std::vector<float> originImageSize{static_cast<float>(origH), static_cast<float>(origW)};
    cv::Mat scaledImg;
    cv::resize(inputImg, scaledImg, cv::Size(Ort::YoloX::IMG_W, Ort::YoloX::IMG_H), 0, 0, cv::INTER_CUBIC);
    osh.preprocess(dst, scaledImg.data, Ort::YoloX::IMG_W, Ort::YoloX::IMG_H, 3);

    clock_t start,end;
    start = clock();
    auto inferenceOutput = osh({dst});
    end = clock();
    std::cout << "Elapsed time in inference: " << (double)(end-start)/CLOCKS_PER_SEC << "s" << std::endl;
    
    // Print shape of output
    // std::cout << "osh outputs is " << inferenceOutput[0].second << "\n"; // osh outputs is 1 8400 85 
    
    std::vector<Ort::YoloX::Object> objects = osh.decodeOutputs(inferenceOutput[0].first, confThresh);
    
    // std::cout << "objects size is " << objects.size() << "\n";
    
    std::vector<std::array<float, 4>> bboxes;
    std::vector<float> scores;
    std::vector<uint64_t> classIndices;

    float scaleW = 1. * origW / Ort::YoloX::IMG_W;
    float scaleH = 1. * origH / Ort::YoloX::IMG_H;

    for (const auto& object : objects) {
        float xmin = object.pos.x * scaleW;
        float ymin = object.pos.y * scaleH;
        float xmax = (object.pos.x + object.pos.width) * scaleW;
        float ymax = (object.pos.y + object.pos.height) * scaleH;

        xmin = std::max<float>(xmin, 0);
        ymin = std::max<float>(ymin, 0);
        xmax = std::min<float>(xmax, inputImg.cols - 1);
        ymax = std::min<float>(ymax, inputImg.rows - 1);

        bboxes.emplace_back(std::array<float, 4>{xmin, ymin, xmax, ymax});
        scores.emplace_back(object.prob);
        classIndices.emplace_back(object.label);

        // Print label and prob
        // std::cout << "object label is " << object.label << "\n";
        // std::cout << "object prob is " << object.prob << "\n";
        
    }

    std::vector<std::array<float, 4>> afterNmsBboxes;
    std::vector<uint64_t> afterNmsClassIndices;
    // try to catch error when bboxes of objects are empty
    try {
        if (!bboxes.empty()) {
            auto afterNmsIndices = Ort::nms(bboxes, scores, confThresh);

            afterNmsBboxes.reserve(afterNmsIndices.size());
            afterNmsClassIndices.reserve(afterNmsIndices.size());

            for (const auto idx : afterNmsIndices) {
                afterNmsBboxes.emplace_back(bboxes[idx]);
                afterNmsClassIndices.emplace_back(classIndices[idx]);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    return afterNmsBboxes.empty()
                    ? inputImg
                    : visualizeOneImage(inputImg, afterNmsBboxes, afterNmsClassIndices, COLORS, osh.classNames());
    
}
}  // namespace
