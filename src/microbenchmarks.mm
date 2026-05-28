#import "mps_infra.h"
#import <Foundation/Foundation.h>

// ===================================================================
// 3.1 Selection — filter l_partkey with threshold
// Raw Metal equivalent: selection_kernel (integer column, 3 thresholds)
// ===================================================================

static void runSingleSelectionTest(id<MTLDevice> device, id<MTLCommandQueue> queue,
                                   MPSGraphTensorData* inputTD, size_t N,
                                   int threshold) {
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* values = [graph placeholderWithShape:@[@(N)]
                                               dataType:MPSDataTypeInt32 name:@"v"];
    MPSGraphTensor* thresh = [graph constantWithScalar:threshold dataType:MPSDataTypeInt32];
    MPSGraphTensor* mask   = [graph lessThanWithPrimaryTensor:values
                                             secondaryTensor:thresh name:@"mask"];
    MPSGraphTensor* maskF  = [graph castTensor:mask toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* count  = [graph reductionSumWithTensor:maskF axis:0 name:@"count"];

    NSDictionary* feeds = @{values: inputTD};

    id<MTLBuffer> outBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* outTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:outTD forKey:count];

    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    float countVal    = *(float*)[outBuf contents];
    float selectivity = 100.0f * countVal / (float)N;
    double bw         = (N * sizeof(int32_t)) / (gpuMs * 1e-3) / 1e9;

    printf("  Filter < %-6d  matches: %8.0f  selectivity: %5.1f%%  GPU: %7.3f ms  BW: %.1f GB/s\n",
           threshold, countVal, selectivity, gpuMs, bw);
}

void runMicroSelection(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- Micro: Selection (l_partkey from lineitem.tbl) ---\n");

    auto t0  = std::chrono::high_resolution_clock::now();
    auto cols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {{1, ColType::INT}});
    auto t1  = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& data = cols.ints(1);
    size_t N   = data.size();
    printf("  Loaded %zu rows (parse: %.1f ms)\n", N, parseMs);

    // Wrap once, reuse across all three threshold tests
    MPSGraphTensorData* inputTD = columnTensor(device, (void*)data.data(), N, MPSDataTypeInt32);

    runSingleSelectionTest(device, queue, inputTD, N, 1000);
    runSingleSelectionTest(device, queue, inputTD, N, 10000);
    runSingleSelectionTest(device, queue, inputTD, N, 50000);
}

// ===================================================================
// 3.2 Aggregation — SUM(l_quantity)
// Raw Metal equivalent: sum_kernel_stage1 + sum_kernel_stage2
// ===================================================================

void runMicroAggregation(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- Micro: Aggregation (SUM l_quantity from lineitem.tbl) ---\n");

    auto t0  = std::chrono::high_resolution_clock::now();
    auto cols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {{4, ColType::FLOAT}});
    auto t1  = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& data = cols.floats(4);
    size_t N   = data.size();
    printf("  Loaded %zu rows (parse: %.1f ms)\n", N, parseMs);

    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* input = [graph placeholderWithShape:@[@(N)]
                                              dataType:MPSDataTypeFloat32 name:@"in"];
    MPSGraphTensor* sum   = [graph reductionSumWithTensor:input axis:0 name:@"sum"];

    MPSGraphTensorData* inputTD = columnTensor(device, (void*)data.data(), N, MPSDataTypeFloat32);
    NSDictionary* feeds = @{input: inputTD};

    id<MTLBuffer> outBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* outTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:outTD forKey:sum];

    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    float sumVal = *(float*)[outBuf contents];
    double bw    = (N * sizeof(float)) / (gpuMs * 1e-3) / 1e9;
    printf("  SUM(l_quantity): %.2f\n", sumVal);
    printf("  GPU time: %.3f ms  Bandwidth: %.1f GB/s\n", gpuMs, bw);
}

// ===================================================================
// 3.3 Join — orders (build) ⋈ lineitem (probe) on orderkey
// Raw Metal equivalent: hash_join_build + hash_join_probe kernels
//
// MPSGraph has no native hash join. Approach used here:
//   Build phase (CPU): mark each orderkey in a membership array.
//   Probe phase (GPU): gather membership[l_orderkey] for every
//                      lineitem row, then sum to count matches.
// This mirrors the hybrid Metal+MPS strategy planned for Q3/Q5/Q9.
// ===================================================================

void runMicroJoin(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- Micro: Join (orders ⋈ lineitem on o_orderkey = l_orderkey) ---\n");

    auto t0 = std::chrono::high_resolution_clock::now();
    auto orderCols = loadColumnsMulti(g_dataset_path + "orders.tbl",  {{0, ColType::INT}});
    auto lineCols  = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {{0, ColType::INT}});
    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& buildKeys = orderCols.ints(0);
    auto& probeKeys = lineCols.ints(0);
    size_t N_build  = buildKeys.size();
    size_t N_probe  = probeKeys.size();
    printf("  Build (orders):   %zu rows\n", N_build);
    printf("  Probe (lineitem): %zu rows\n", N_probe);
    printf("  Parse time: %.1f ms\n", parseMs);

    // CPU build phase: boolean membership array indexed by orderkey
    auto tb0 = std::chrono::high_resolution_clock::now();
    int maxKey = *std::max_element(buildKeys.begin(), buildKeys.end());
    std::vector<float> membership((size_t)(maxKey + 1), 0.0f);
    for (int k : buildKeys) membership[(size_t)k] = 1.0f;
    auto tb1 = std::chrono::high_resolution_clock::now();
    double buildMs = std::chrono::duration<double, std::milli>(tb1 - tb0).count();
    printf("  CPU build (membership array, maxKey=%d): %.1f ms\n", maxKey, buildMs);

    // GPU probe phase: gather + sum
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* mem  = [graph placeholderWithShape:@[@(maxKey + 1)]
                                             dataType:MPSDataTypeFloat32 name:@"mem"];
    MPSGraphTensor* keys = [graph placeholderWithShape:@[@(N_probe)]
                                             dataType:MPSDataTypeInt32  name:@"keys"];

    MPSGraphTensor* gathered   = [graph gatherWithUpdatesTensor:mem
                                                  indicesTensor:keys
                                                           axis:0
                                                batchDimensions:0
                                                           name:@"gathered"];
    MPSGraphTensor* matchCount = [graph reductionSumWithTensor:gathered axis:0 name:@"count"];

    MPSGraphTensorData* memTD  = columnTensor(device, membership.data(),
                                              (size_t)(maxKey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* keysTD = columnTensor(device, (void*)probeKeys.data(),
                                              N_probe, MPSDataTypeInt32);
    NSDictionary* feeds = @{mem: memTD, keys: keysTD};

    id<MTLBuffer> outBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* outTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:outTD
                                                                       forKey:matchCount];

    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    float matches = *(float*)[outBuf contents];
    double bw = (N_probe * sizeof(int32_t) + (size_t)(maxKey + 1) * sizeof(float))
                / (gpuMs * 1e-3) / 1e9;
    printf("  Matches: %.0f\n", matches);
    printf("  GPU probe time: %.3f ms  Bandwidth: %.1f GB/s\n", gpuMs, bw);
    printf("  Total (CPU build + GPU probe): %.3f ms\n", buildMs + gpuMs);
}

// ===================================================================
// Entry point — called from main.mm
// ===================================================================

void runMicrobenchmarks(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n=== MPS Micro-benchmarks ===\n");
    runMicroAggregation(device, queue);
    runMicroSelection(device, queue);
    runMicroJoin(device, queue);
    printf("\n=== Micro-benchmarks complete ===\n");
}
