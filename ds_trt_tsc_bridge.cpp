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
    static uint32_t bufferCount = 0, profileFrameCount = 0;
    static torch::ScalarType sourceType = torch::kFloat32;

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

    if(outputLayerIndex == -1 && batchDim > 0) {
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

    if(dims[0] != batchDim) {
        nvtxRangePop(); // setup
        // onnx export cannot handle dynamic batch dimension just yet
        return true;
    }

    at::Tensor source_nchw = torch::from_blob(
        layer.buffer,
        c10::IntArrayRef(dims),
        at::device(torch::kCUDA).dtype(sourceType)
    );

    at::Tensor batch_nchw = source_nchw.to(torch::kCUDA, torch::kFloat32, false, true).contiguous();

    nvtxRangePop(); // setup

/*         if(bufferCount == 57) { */
/*             at::Tensor batch_cpu = batch_nchw.to(torch::kCPU); */
/*             /1* std::cout << "batch bytes min/max/mean:\n" << batch_cpu.min() << "\n" << batch_cpu.max() << "\n" << batch_cpu.to(torch::kFloat32).mean() << "\n"; *1/ */
/*             /1* std::cout << batch_cpu.sizes() << "\n"; *1/ */
/*             for(unsigned int i = 0; i < batch_cpu.sizes()[0]; ++i) { */
/*                 ppm_save(batch_cpu.slice(0, i, i + 1).squeeze(0), "logs/test_" + std::to_string(bufferCount) + "_" + std::to_string(i) + ".ppm"); */
/*             } */
/*         } */

    nvtxRangePushA("inference");
    auto result = model.forward(
        std::vector<torch::jit::IValue>({batch_nchw})
    ).toTuple();
    nvtxRangePop(); // inference

    std::cout << "[" << bufferCount << "]:\tdetections: " << result->elements()[0].toTensor().sizes()[0] << "\n";

    if(profileFrameCount >= fpsFramePeriod) {
        std::chrono::duration<double> elapsed_s = std::chrono::system_clock::now() - start;
        std::cout << profileFrameCount / elapsed_s.count() << " FPS\n";
        profileFrameCount = 0;
        start = std::chrono::system_clock::now();
    }

    bufferCount += 1;
    profileFrameCount += batchDim;
    return true;
}


