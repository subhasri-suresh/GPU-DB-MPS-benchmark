#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q14 — Promotion Effect
//
// SELECT 100.00 * SUM(CASE WHEN p_type LIKE 'PROMO%'
//                          THEN l_extendedprice * (1 - l_discount) ELSE 0 END)
//        / SUM(l_extendedprice * (1 - l_discount)) AS promo_revenue
// FROM lineitem, part
// WHERE l_partkey = p_partkey
//   AND l_shipdate >= '1995-09-01'
//   AND l_shipdate <  '1995-10-01';
//
// Strategy:
//   Build phase (CPU): scan part.tbl, mark each partkey as promo (1.0) or not (0.0).
//   Probe phase (GPU): gather promo flag per lineitem row via l_partkey index,
//                      apply date filter, then run two parallel reductionSums.
//   Post (CPU):        divide promo_sum / total_sum for the percentage.

void runQ14(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q14: Promotion Effect ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto tStart = std::chrono::high_resolution_clock::now();

    // part.tbl — p_partkey (col 0), p_type (col 4, 25-char fixed)
    auto partCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT},
        {4, ColType::CHAR_FIXED, 25},
    });
    auto& p_partkey    = partCols.ints(0);
    auto& p_type_chars = partCols.chars(4);
    size_t N_part = p_partkey.size();

    // lineitem.tbl — l_partkey, l_extendedprice, l_discount, l_shipdate
    auto lineCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 1, ColType::INT  },
        { 5, ColType::FLOAT},
        { 6, ColType::FLOAT},
        {10, ColType::DATE },
    });
    auto& l_partkey  = lineCols.ints(1);
    auto& l_price    = lineCols.floats(5);
    auto& l_discount = lineCols.floats(6);
    auto& l_shipdate = lineCols.ints(10);
    size_t N = l_partkey.size();

    auto tParsed = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(tParsed - tStart).count();
    printf("  Part rows: %zu  Lineitem rows: %zu  (parse: %.1f ms)\n", N_part, N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU build — promo membership array
    //   promoMembership[partkey] = 1.0 if p_type starts with "PROMO"
    // ----------------------------------------------------------------
    int maxPartkey = *std::max_element(p_partkey.begin(), p_partkey.end());
    std::vector<float> promoMembership((size_t)(maxPartkey + 1), 0.0f);
    for (size_t i = 0; i < N_part; i++) {
        if (strncmp(&p_type_chars[i * 25], "PROMO", 5) == 0)
            promoMembership[(size_t)p_partkey[i]] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tPromo    = [graph placeholderWithShape:@[@(maxPartkey + 1)]
                                                  dataType:MPSDataTypeFloat32 name:@"promo"];
    MPSGraphTensor* tLPartkey = [graph placeholderWithShape:@[@(N)]
                                                  dataType:MPSDataTypeInt32   name:@"l_partkey"];
    MPSGraphTensor* tPrice    = [graph placeholderWithShape:@[@(N)]
                                                  dataType:MPSDataTypeFloat32 name:@"price"];
    MPSGraphTensor* tDiscount = [graph placeholderWithShape:@[@(N)]
                                                  dataType:MPSDataTypeFloat32 name:@"discount"];
    MPSGraphTensor* tShipdate = [graph placeholderWithShape:@[@(N)]
                                                  dataType:MPSDataTypeInt32   name:@"shipdate"];

    // Date filter: 19950901 <= l_shipdate < 19951001
    MPSGraphTensor* d1     = [graph constantWithScalar:19950901 dataType:MPSDataTypeInt32];
    MPSGraphTensor* d2     = [graph constantWithScalar:19951001 dataType:MPSDataTypeInt32];
    MPSGraphTensor* cDate1 = [graph greaterThanOrEqualToWithPrimaryTensor:tShipdate
                                                         secondaryTensor:d1 name:nil];
    MPSGraphTensor* cDate2 = [graph lessThanWithPrimaryTensor:tShipdate
                                              secondaryTensor:d2 name:nil];
    MPSGraphTensor* dateMask  = [graph logicalANDWithPrimaryTensor:cDate1
                                                  secondaryTensor:cDate2 name:@"dateMask"];
    MPSGraphTensor* dateMaskF = [graph castTensor:dateMask toType:MPSDataTypeFloat32 name:nil];

    // Gather promo flag for each lineitem row using l_partkey as index
    MPSGraphTensor* promoFlag = [graph gatherWithUpdatesTensor:tPromo
                                                 indicesTensor:tLPartkey
                                                          axis:0
                                               batchDimensions:0
                                                          name:@"promoFlag"];

    // disc_price = extendedprice * (1 - discount)
    MPSGraphTensor* one       = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* oneMinusD = [graph subtractionWithPrimaryTensor:one
                                                   secondaryTensor:tDiscount name:nil];
    MPSGraphTensor* discPrice = [graph multiplicationWithPrimaryTensor:tPrice
                                                      secondaryTensor:oneMinusD name:nil];

    // Apply date mask → rows outside the date window become 0
    MPSGraphTensor* maskedDiscPrice = [graph multiplicationWithPrimaryTensor:discPrice
                                                            secondaryTensor:dateMaskF name:nil];

    // Promo contribution = masked disc_price * promo flag
    MPSGraphTensor* promoContrib = [graph multiplicationWithPrimaryTensor:maskedDiscPrice
                                                         secondaryTensor:promoFlag name:nil];

    // Two parallel reductions — run in a single graph execution
    MPSGraphTensor* promoSum = [graph reductionSumWithTensor:promoContrib   axis:0 name:@"promoSum"];
    MPSGraphTensor* totalSum = [graph reductionSumWithTensor:maskedDiscPrice axis:0 name:@"totalSum"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffers
    // ----------------------------------------------------------------
    MPSGraphTensorData* promTD = columnTensor(device, promoMembership.data(),
                                              (size_t)(maxPartkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* pkTD   = columnTensor(device, (void*)l_partkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* prTD   = columnTensor(device, (void*)l_price.data(),    N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD   = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* sdTD   = columnTensor(device, (void*)l_shipdate.data(), N, MPSDataTypeInt32);

    NSDictionary* feeds = @{
        tPromo:    promTD,
        tLPartkey: pkTD,
        tPrice:    prTD,
        tDiscount: diTD,
        tShipdate: sdTD
    };

    id<MTLBuffer> promoBuf = allocSharedBuffer(device, sizeof(float));
    id<MTLBuffer> totalBuf = allocSharedBuffer(device, sizeof(float));

    MPSGraphTensorData* promoOutTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:promoBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* totalOutTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:totalBuf shape:@[@1] dataType:MPSDataTypeFloat32];

    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[promoSum] = promoOutTD;
    results[totalSum] = totalOutTD;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup runs + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — divide promo_sum / total_sum
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float pSum = *(float*)[promoBuf contents];
    float tSum = *(float*)[totalBuf contents];
    double promoRevenue = (tSum > 0.0f) ? 100.0 * (double)pSum / (double)tSum : 0.0;
    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("  promo_revenue: %.4f%%\n", promoRevenue);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q14", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
