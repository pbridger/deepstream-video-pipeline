#include <torch/script.h>
#include <torchvision/nms.h>
#include <c10/cuda/CUDAGuard.h>
#include <nvToolsExt.h>

#include "nvdsinfer.h"
#include "nvdsinfer_custom_impl.h"

#include <iostream>
#include <memory>
#include <chrono>

auto model = torch::jit::load(std::getenv("DS_TSC_PTH_PATH"));

const size_t numStreams = 3;


size_t ppm_save(const at::Tensor& image_chw, const std::string& filename) {
    FILE* outfile = fopen(filename.c_str(), "w");

    at::Tensor bytes_hwc = image_chw.permute({1, 2, 0}).to(torch::kCPU, torch::kUInt8, false, true).contiguous();
    unsigned int height = bytes_hwc.sizes()[0], width = bytes_hwc.sizes()[1];

    size_t n = 0;
    n += fprintf(outfile, "P6\n# THIS IS A COMMENT\n%d %d\n%d\n", width, height, 0xFF);
    n += fwrite((uint8_t*)bytes_hwc.data_ptr(), 1, width * height * 3, outfile);

    fclose(outfile);
    return n;
}


extern "C"
bool DsTrtTscBridgeDevice(
    std::vector<std::vector<NvDsInferLayerInfo>> const &batchOutputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<std::vector<NvDsInferObjectDetectionInfo>> &batchObjectList
);

extern "C"
bool DsTrtTscBridgeDevice(
    std::vector<std::vector<NvDsInferLayerInfo>> const &batchOutputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<std::vector<NvDsInferObjectDetectionInfo>> &batchObjectList
) {
    static size_t outputLayerIndex = -1;
    const uint32_t fpsFramePeriod = 128;
    static uint32_t bufferCount = 0, profileFrameCount = 0, nextFpsFrameCount = fpsFramePeriod;
    static torch::ScalarType sourceType = torch::kFloat32;

    std::cout << "[" << bufferCount << "]\n";

    static std::vector<at::cuda::CUDAStream> cudaStreams;
    while(cudaStreams.size() < numStreams) {
        cudaStreams.emplace_back(at::cuda::getStreamFromPool());
    }

    static auto start = std::chrono::system_clock::now();
    if(bufferCount == 5) {
        start = std::chrono::system_clock::now();
        profileFrameCount = 0;
    }

    at::cuda::CUDAStreamGuard streamGuard(cudaStreams[bufferCount % cudaStreams.size()]);
    nvtxRangePushA("setup");

    unsigned int batchDim = batchOutputLayersInfo.size();
    std::cout << "batch size: " << batchDim << "\n";

    if(outputLayerIndex == -1 && batchDim > 0) {
        std::cout << "Moving model to device\n";
        model.to(torch::kCUDA);

        outputLayerIndex = 0;
        auto& layer = batchOutputLayersInfo[0][outputLayerIndex];

        sourceType = torch::kFloat32;
        if(layer.dataType == NvDsInferDataType::HALF) {
            sourceType = torch::kFloat16;
        } else if(layer.dataType == NvDsInferDataType::INT8) {
            sourceType = torch::kUInt8;
        }
    }

    auto& layer = batchOutputLayersInfo[0][outputLayerIndex];
    std::vector<int64_t> dims;
    for(unsigned int d = 0; d < layer.inferDims.numDims; ++d) dims.push_back(layer.inferDims.d[d]);
    std::cout << "dims: " << dims << "\n";

    if(dims[0] != batchDim) {
        nvtxRangePop(); // setup
        // onnx export cannot handle dynamic batch dimension just yet
        return true;
    }

    at::Tensor source_nchw = torch::from_blob(
        layer.buffer,
        c10::IntArrayRef(dims),
        at::dtype(sourceType)
    );

    at::Tensor batch_nchw = source_nchw.to(torch::kCUDA, torch::kFloat32, false, true).contiguous();

    nvtxRangePop(); // setup

    nvtxRangePushA("inference");
    auto result = model.forward(
        std::vector<torch::jit::IValue>({batch_nchw})
    ).toTuple();
    nvtxRangePop(); // inference

    std::cout << "[" << bufferCount << "]:\tdetections: " << result->elements()[0].toTensor().sizes()[0] << "\n";

    if(profileFrameCount >= nextFpsFrameCount) {
        std::chrono::duration<double> elapsed_s = std::chrono::system_clock::now() - start;
        std::cout << profileFrameCount / elapsed_s.count() << " FPS\n";
        nextFpsFrameCount += fpsFramePeriod;
    }

    bufferCount += 1;
    profileFrameCount += batchDim;
    return true;
}


