#import "mps_infra.h"

// Global dataset configuration (defined once here)
std::string g_dataset_path = "data/SF-1/";
bool g_sf100_mode = false;

// Returns the byte size for an MPSDataType
static size_t dtypeSize(MPSDataType dtype) {
    switch (dtype) {
        case MPSDataTypeFloat32: return 4;
        case MPSDataTypeInt32:   return 4;
        case MPSDataTypeFloat16: return 2;
        case MPSDataTypeInt16:   return 2;
        case MPSDataTypeInt8:    return 1;
        case MPSDataTypeUInt8:   return 1;
        default:                 return 4;
    }
}

MPSGraphTensorData* columnTensor(id<MTLDevice> device,
                                  void* data, size_t count,
                                  MPSDataType dtype) {
    size_t bytes = count * dtypeSize(dtype);
    id<MTLBuffer> buf = [device newBufferWithBytesNoCopy:data
                                                   length:bytes
                                                  options:MTLResourceStorageModeShared
                                              deallocator:nil];
    MPSGraphTensorData* td = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:buf
                    shape:@[@(count)]
                 dataType:dtype];
    return td;
}

id<MTLBuffer> allocSharedBuffer(id<MTLDevice> device, size_t bytes) {
    return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

double runGraphTimed(MPSGraph* graph,
                     NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                     NSArray<MPSGraphTensor*>* targets,
                     id<MTLCommandQueue> queue,
                     NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results) {
    auto wallT0 = std::chrono::high_resolution_clock::now();

    if (results) {
        // Synchronous: blocks until done, writes directly into pre-allocated output buffers.
        [graph runWithMTLCommandQueue:queue
                                feeds:feeds
                     targetOperations:nil
                    resultsDictionary:results];
    } else {
        // Warmup path: compute output tensors without storing results.
        [graph runWithMTLCommandQueue:queue
                                feeds:feeds
                        targetTensors:targets
                     targetOperations:nil];
    }

    auto wallT1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(wallT1 - wallT0).count();
}

double runQueryWarmedUp(MPSGraph* graph,
                        NSDictionary<MPSGraphTensor*, MPSGraphTensorData*>* feeds,
                        NSArray<MPSGraphTensor*>* targets,
                        id<MTLCommandQueue> queue,
                        NSMutableDictionary<MPSGraphTensor*, MPSGraphTensorData*>* results) {
    NSArray<MPSGraphTensor*>* warmupTargets = targets ?: [results allKeys];

    double gpuSec = 0.0;
    for (int iter = 0; iter < 3; iter++) {
        if (iter < 2)
            gpuSec = runGraphTimed(graph, feeds, warmupTargets, queue, nil);
        else
            gpuSec = runGraphTimed(graph, feeds, targets, queue, results);
    }
    return gpuSec * 1000.0; // convert to ms
}
