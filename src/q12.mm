#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q12 — Shipping Modes and Order Priority
//
// SELECT l_shipmode,
//   SUM(CASE WHEN o_orderpriority IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) high_line_count,
//   SUM(CASE WHEN o_orderpriority NOT IN ('1-URGENT','2-HIGH') THEN 1 ELSE 0 END) low_line_count
// FROM orders, lineitem
// WHERE o_orderkey = l_orderkey
//   AND l_shipmode IN ('MAIL','SHIP')
//   AND l_commitdate < l_receiptdate
//   AND l_shipdate   < l_commitdate
//   AND l_receiptdate >= '1994-01-01' AND l_receiptdate < '1995-01-01'
// GROUP BY l_shipmode ORDER BY l_shipmode;
//
// Strategy:
//   Build (CPU): orders → highPriority[orderkey] = 1.0 if priority char '1' or '2'.
//   CPU:         lineitem → groupIdx (0=MAIL, 1=SHIP), shipmodeMask (1.0 if MAIL/SHIP).
//   GPU:         date filter conditions, gather priority, scatter into 2 bins.

void runQ12(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q12: Shipping Modes and Order Priority ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    // orders: o_orderkey (col 0), o_orderpriority first char (col 5)
    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT  },   // o_orderkey
        {5, ColType::CHAR1},   // o_orderpriority (first char: '1'=URGENT, '2'=HIGH)
    });
    auto& o_orderkey  = oCols.ints(0);
    auto& o_priority  = oCols.chars(5);
    size_t N_orders   = o_orderkey.size();

    // lineitem: l_orderkey, l_shipdate, l_commitdate, l_receiptdate, l_shipmode (first char)
    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 0, ColType::INT  },   // l_orderkey
        {10, ColType::DATE },   // l_shipdate
        {11, ColType::DATE },   // l_commitdate
        {12, ColType::DATE },   // l_receiptdate
        {14, ColType::CHAR1},   // l_shipmode first char: 'M'=MAIL, 'S'=SHIP
    });
    auto& l_orderkey    = lCols.ints(0);
    auto& l_shipdate    = lCols.ints(10);
    auto& l_commitdate  = lCols.ints(11);
    auto& l_receiptdate = lCols.ints(12);
    auto& l_shipmode    = lCols.chars(14);
    size_t N = l_orderkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n", N_orders, N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU build — highPriority membership array
    //   highPriority[orderkey] = 1.0 if priority first char is '1' or '2'
    // ----------------------------------------------------------------
    int maxOrderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());
    std::vector<float> highPriority((size_t)(maxOrderkey + 1), 0.0f);
    for (size_t i = 0; i < N_orders; i++) {
        char p = o_priority[i];
        if (p == '1' || p == '2')
            highPriority[(size_t)o_orderkey[i]] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — compute groupIdx and shipmodeMask per lineitem row
    //   'M' (MAIL) → group 0,  'S' (SHIP) → group 1,  other → group 0 + mask=0
    // ----------------------------------------------------------------
    std::vector<int32_t> groupIdx(N, 0);
    std::vector<float>   shipmodeMask(N, 0.0f);
    for (size_t i = 0; i < N; i++) {
        char sm = l_shipmode[i];
        if (sm == 'M') { groupIdx[i] = 0; shipmodeMask[i] = 1.0f; }
        else if (sm == 'S') { groupIdx[i] = 1; shipmodeMask[i] = 1.0f; }
    }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tHighPri   = [graph placeholderWithShape:@[@(maxOrderkey + 1)]
                                                    dataType:MPSDataTypeFloat32 name:@"highPri"];
    MPSGraphTensor* tOrderkey  = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tShipdate  = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"shipdate"];
    MPSGraphTensor* tCommit    = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"commitdate"];
    MPSGraphTensor* tReceipt   = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"receiptdate"];
    MPSGraphTensor* tSMMask    = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeFloat32 name:@"shipmodeMask"];
    MPSGraphTensor* tGroupIdx  = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"groupIdx"];

    // Date conditions
    MPSGraphTensor* rStart  = [graph constantWithScalar:19940101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* rEnd    = [graph constantWithScalar:19950101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* cR1     = [graph greaterThanOrEqualToWithPrimaryTensor:tReceipt
                                                           secondaryTensor:rStart name:nil];
    MPSGraphTensor* cR2     = [graph lessThanWithPrimaryTensor:tReceipt
                                               secondaryTensor:rEnd name:nil];
    MPSGraphTensor* cCD     = [graph lessThanWithPrimaryTensor:tCommit
                                               secondaryTensor:tReceipt name:nil];  // commitdate < receiptdate
    MPSGraphTensor* cSD     = [graph lessThanWithPrimaryTensor:tShipdate
                                               secondaryTensor:tCommit name:nil];   // shipdate < commitdate

    MPSGraphTensor* dateMask = [graph logicalANDWithPrimaryTensor:
                                    [graph logicalANDWithPrimaryTensor:
                                        [graph logicalANDWithPrimaryTensor:cR1 secondaryTensor:cR2 name:nil]
                                        secondaryTensor:cCD name:nil]
                                    secondaryTensor:cSD name:@"dateMask"];
    MPSGraphTensor* dateMaskF = [graph castTensor:dateMask toType:MPSDataTypeFloat32 name:nil];

    // Combined filter = date mask × shipmode mask
    MPSGraphTensor* combinedMask = [graph multiplicationWithPrimaryTensor:dateMaskF
                                                         secondaryTensor:tSMMask name:@"combinedMask"];

    // Gather high-priority flag per lineitem row
    MPSGraphTensor* highFlag = [graph gatherWithUpdatesTensor:tHighPri
                                                indicesTensor:tOrderkey
                                                         axis:0
                                              batchDimensions:0
                                                         name:@"highFlag"];

    // high_count contribution = combinedMask * highFlag
    // low_count  contribution = combinedMask * (1 - highFlag)
    MPSGraphTensor* one      = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* lowFlag  = [graph subtractionWithPrimaryTensor:one
                                                  secondaryTensor:highFlag name:nil];
    MPSGraphTensor* highContr = [graph multiplicationWithPrimaryTensor:combinedMask
                                                      secondaryTensor:highFlag name:nil];
    MPSGraphTensor* lowContr  = [graph multiplicationWithPrimaryTensor:combinedMask
                                                      secondaryTensor:lowFlag  name:nil];

    // Scatter into 2-bin accumulators (MAIL=0, SHIP=1)
    MPSGraphTensor* zeros2 = [graph constantWithScalar:0.0 shape:@[@2]
                                              dataType:MPSDataTypeFloat32];
    MPSGraphTensor* highCount = [graph scatterWithDataTensor:zeros2 updatesTensor:highContr
                                               indicesTensor:tGroupIdx axis:0
                                                        mode:MPSGraphScatterModeAdd name:@"highCount"];
    MPSGraphTensor* lowCount  = [graph scatterWithDataTensor:zeros2 updatesTensor:lowContr
                                               indicesTensor:tGroupIdx axis:0
                                                        mode:MPSGraphScatterModeAdd name:@"lowCount"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + outputs
    // ----------------------------------------------------------------
    MPSGraphTensorData* hpTD  = columnTensor(device, highPriority.data(),
                                             (size_t)(maxOrderkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* okTD  = columnTensor(device, (void*)l_orderkey.data(),    N, MPSDataTypeInt32);
    MPSGraphTensorData* sdTD  = columnTensor(device, (void*)l_shipdate.data(),    N, MPSDataTypeInt32);
    MPSGraphTensorData* cdTD  = columnTensor(device, (void*)l_commitdate.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* rdTD  = columnTensor(device, (void*)l_receiptdate.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* smTD  = columnTensor(device, shipmodeMask.data(),         N, MPSDataTypeFloat32);
    MPSGraphTensorData* giTD  = columnTensor(device, groupIdx.data(),             N, MPSDataTypeInt32);

    NSDictionary* feeds = @{
        tHighPri:  hpTD, tOrderkey: okTD, tShipdate: sdTD,
        tCommit:   cdTD, tReceipt:  rdTD, tSMMask:   smTD,
        tGroupIdx: giTD
    };

    id<MTLBuffer> highBuf = allocSharedBuffer(device, 2 * sizeof(float));
    id<MTLBuffer> lowBuf  = allocSharedBuffer(device, 2 * sizeof(float));
    MPSGraphTensorData* highOutTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:highBuf shape:@[@2] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* lowOutTD  = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:lowBuf  shape:@[@2] dataType:MPSDataTypeFloat32];

    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[highCount] = highOutTD;
    results[lowCount]  = lowOutTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: Print results
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* high = (float*)[highBuf contents];
    float* low  = (float*)[lowBuf  contents];
    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q12 Results:\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| shipmode | high_line_count  | low_line_count  |\n");
    printf("+----------+------------------+-----------------+\n");
    printf("| MAIL     | %16.0f | %15.0f |\n", high[0], low[0]);
    printf("| SHIP     | %16.0f | %15.0f |\n", high[1], low[1]);
    printf("+----------+------------------+-----------------+\n");

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q12", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
