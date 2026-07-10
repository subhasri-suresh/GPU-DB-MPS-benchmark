#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q4 — Order Priority Checking
//
// SELECT o_orderpriority, COUNT(*) AS order_count
// FROM orders
// WHERE o_orderdate >= DATE '1993-07-01'
//   AND o_orderdate <  DATE '1993-07-01' + INTERVAL '3' MONTH
//   AND EXISTS (SELECT * FROM lineitem
//               WHERE l_orderkey = o_orderkey AND l_commitdate < l_receiptdate)
// GROUP BY o_orderpriority ORDER BY o_orderpriority;
//
// Strategy:
//   CPU build: "late" map indexed by orderkey — 1.0 if any of its lineitems
//              has l_commitdate < l_receiptdate (implements the EXISTS semi-join).
//   MPSGraph:  gather late-map per order, AND with date filter, group by
//              orderpriority digit (5 fixed buckets) via masked reductionSum.

static constexpr int NUM_PRIORITIES = 5;

void runQ4(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q4: Order Priority Checking ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 0, ColType::INT },   // l_orderkey
        {11, ColType::DATE},   // l_commitdate
        {12, ColType::DATE},   // l_receiptdate
    });
    auto& l_orderkey   = lCols.ints(0);
    auto& l_commitdate = lCols.ints(11);
    auto& l_receiptdate = lCols.ints(12);

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT  },   // o_orderkey
        {4, ColType::DATE },   // o_orderdate
        {5, ColType::CHAR1},   // o_orderpriority (first char is the priority digit)
    });
    auto& o_orderkey     = oCols.ints(0);
    auto& o_orderdate    = oCols.ints(4);
    auto& o_orderpriority = oCols.chars(5);
    size_t M = o_orderkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Lineitem: %zu  Orders: %zu  (parse: %.1f ms)\n", l_orderkey.size(), M, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — late-order map (orderkey -> 1.0 if EXISTS late lineitem)
    // ----------------------------------------------------------------
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());
    std::vector<float> late_map((size_t)(max_orderkey + 1), 0.0f);
    for (size_t i = 0; i < l_orderkey.size(); i++) {
        int ok = l_orderkey[i];
        if (ok >= 0 && ok <= max_orderkey && l_commitdate[i] < l_receiptdate[i])
            late_map[(size_t)ok] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — group_idx per order (priority digit '1'..'5' -> 0..4)
    // ----------------------------------------------------------------
    std::vector<int32_t> groupIdx(M);
    for (size_t i = 0; i < M; i++)
        groupIdx[i] = o_orderpriority[i] - '1';

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tLateMap  = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                    dataType:MPSDataTypeFloat32 name:@"late_map"];
    MPSGraphTensor* tOrdKey   = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeInt32   name:@"o_orderkey"];
    MPSGraphTensor* tOrdDate  = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeInt32   name:@"o_orderdate"];
    MPSGraphTensor* tGroupIdx = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeInt32   name:@"group_idx"];

    // EXISTS semi-join: gather late flag per order
    MPSGraphTensor* lateMatch = [graph gatherWithUpdatesTensor:tLateMap
                                                  indicesTensor:tOrdKey
                                                           axis:0 batchDimensions:0 name:@"late_match"];
    MPSGraphTensor* halfF     = [graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* lateBool  = [graph greaterThanWithPrimaryTensor:lateMatch secondaryTensor:halfF name:nil];

    // Date filter: [1993-07-01, 1993-10-01)
    MPSGraphTensor* startDate = [graph constantWithScalar:19930701 dataType:MPSDataTypeInt32];
    MPSGraphTensor* endDate   = [graph constantWithScalar:19931001 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateOk    = [graph logicalANDWithPrimaryTensor:
                                    [graph greaterThanOrEqualToWithPrimaryTensor:tOrdDate
                                                                 secondaryTensor:startDate name:nil]
                                 secondaryTensor:
                                    [graph lessThanWithPrimaryTensor:tOrdDate
                                                     secondaryTensor:endDate name:nil]
                                 name:nil];

    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:dateOk secondaryTensor:lateBool name:nil]
                                     toType:MPSDataTypeFloat32 name:@"mask"];

    // Per-priority masked count via reductionSum
    NSMutableArray* gCnt = [@[] mutableCopy];
    for (int g = 0; g < NUM_PRIORITIES; g++) {
        MPSGraphTensor* gConst = [graph constantWithScalar:g dataType:MPSDataTypeInt32];
        MPSGraphTensor* gMaskF = [graph castTensor:
                                     [graph equalWithPrimaryTensor:tGroupIdx secondaryTensor:gConst name:nil]
                                         toType:MPSDataTypeFloat32 name:nil];
        [gCnt addObject:[graph reductionSumWithTensor:
                             [graph multiplicationWithPrimaryTensor:maskF secondaryTensor:gMaskF name:nil]
                         axis:0 name:nil]];
    }
    MPSGraphTensor* outCnt = [graph concatTensors:gCnt dimension:0 name:@"counts"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* lmTD = columnTensor(device, late_map.data(), (size_t)(max_orderkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* okTD = columnTensor(device, (void*)o_orderkey.data(),  M, MPSDataTypeInt32);
    MPSGraphTensorData* odTD = columnTensor(device, (void*)o_orderdate.data(), M, MPSDataTypeInt32);
    MPSGraphTensorData* giTD = columnTensor(device, groupIdx.data(),           M, MPSDataTypeInt32);

    NSDictionary* feeds = @{
        tLateMap: lmTD, tOrdKey: okTD, tOrdDate: odTD, tGroupIdx: giTD
    };

    id<MTLBuffer> cntBuf = allocSharedBuffer(device, NUM_PRIORITIES * sizeof(float));
    MPSGraphTensorData* cntTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:cntBuf shape:@[@(NUM_PRIORITIES)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[outCnt] = cntTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: CPU post — print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* rCnt = (float*)[cntBuf contents];

    static const char* prio_names[] = {"1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"};
    printf("\nTPC-H Q4 Results:\n");
    printf("+------------------+-------------+\n");
    printf("| o_orderpriority  | order_count |\n");
    printf("+------------------+-------------+\n");
    for (int i = 0; i < NUM_PRIORITIES; i++)
        printf("| %-16s | %11.0f |\n", prio_names[i], rCnt[i]);
    printf("+------------------+-------------+\n");

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", M);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q4", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
