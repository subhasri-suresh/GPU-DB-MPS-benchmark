#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q22 — Global Sales Opportunity
//
// SELECT cntrycode, COUNT(*) AS numcust, SUM(c_acctbal) AS totacctbal
// FROM (
//   SELECT substring(c_phone,1,2) AS cntrycode, c_acctbal
//   FROM customer
//   WHERE substring(c_phone,1,2) IN ('13','31','23','29','30','18','17')
//     AND c_acctbal > (
//         SELECT AVG(c_acctbal) FROM customer
//         WHERE c_acctbal > 0.00 AND substring(c_phone,1,2) IN ('13','31','23','29','30','18','17'))
//     AND NOT EXISTS (SELECT * FROM orders WHERE o_custkey = c_custkey)
// ) AS custsale
// GROUP BY cntrycode ORDER BY cntrycode;
//
// Strategy (fully GPU, two tables in one graph):
//   CPU build: cntrycode index (0..6, or -1) per customer from the phone prefix (string op).
//   MPSGraph:  pass 1 — scatter-add ones over orders indexed by o_custkey gives an
//              order-count-per-customer array; gathering it back per customer row is the
//              NOT EXISTS check (count==0). pass 2 — masked reductionSum/count over
//              customer gives the correlated AVG(c_acctbal) subquery. Both live in the
//              same graph/command buffer; per-country masked sums close out the query.

static constexpr int NUM_CODES = 7;
static const char* COUNTRY_CODES[NUM_CODES] = {"13", "31", "23", "29", "30", "18", "17"};

void runQ22(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q22: Global Sales Opportunity ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT        },
        {4, ColType::CHAR_FIXED, 15},   // c_phone
        {5, ColType::FLOAT      },      // c_acctbal
    });
    auto& c_custkey = cCols.ints(0);
    auto& c_phone   = cCols.chars(4);
    auto& c_acctbal = cCols.floats(5);
    size_t C = c_custkey.size();

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {{1, ColType::INT}});
    auto& o_custkey = oCols.ints(1);
    size_t M = o_custkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Customer: %zu  Orders: %zu  (parse: %.1f ms)\n", C, M, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — cntrycode membership + index, per customer
    // ----------------------------------------------------------------
    std::vector<float>   inSet(C, 0.0f);
    std::vector<int32_t> countryIdx(C, 0);
    for (size_t i = 0; i < C; i++) {
        std::string prefix(c_phone.data() + i * 15, 2);
        for (int k = 0; k < NUM_CODES; k++) {
            if (prefix == COUNTRY_CODES[k]) { inSet[i] = 1.0f; countryIdx[i] = k; break; }
        }
    }
    int max_custkey = *std::max_element(c_custkey.begin(), c_custkey.end());

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    // --- orders subgraph: order count per customer ---
    MPSGraphTensor* tOCustkey = [graph placeholderWithShape:@[@(M)] dataType:MPSDataTypeInt32 name:@"o_custkey"];
    MPSGraphTensor* onesM = [graph constantWithScalar:1.0 shape:@[@(M)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* zerosCust = [graph constantWithScalar:0.0 shape:@[@(max_custkey + 1)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* orderCount = [graph scatterWithDataTensor:zerosCust
                                                updatesTensor:onesM
                                                indicesTensor:tOCustkey
                                                         axis:0
                                                         mode:MPSGraphScatterModeAdd
                                                         name:@"order_count"];

    // --- customer subgraph ---
    MPSGraphTensor* tCustkey    = [graph placeholderWithShape:@[@(C)] dataType:MPSDataTypeInt32   name:@"c_custkey"];
    MPSGraphTensor* tAcctbal    = [graph placeholderWithShape:@[@(C)] dataType:MPSDataTypeFloat32 name:@"c_acctbal"];
    MPSGraphTensor* tInSet      = [graph placeholderWithShape:@[@(C)] dataType:MPSDataTypeFloat32 name:@"in_set"];
    MPSGraphTensor* tCountryIdx = [graph placeholderWithShape:@[@(C)] dataType:MPSDataTypeInt32   name:@"country_idx"];

    MPSGraphTensor* zeroF = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* halfF = [graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32];

    // Correlated AVG(c_acctbal) over {c_acctbal>0 AND in_set}
    MPSGraphTensor* posBalF = [graph castTensor:
                                   [graph greaterThanWithPrimaryTensor:tAcctbal secondaryTensor:zeroF name:nil]
                                       toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* avgMaskF = [graph multiplicationWithPrimaryTensor:tInSet secondaryTensor:posBalF name:nil];
    MPSGraphTensor* avgNum = [graph reductionSumWithTensor:
                                  [graph multiplicationWithPrimaryTensor:tAcctbal secondaryTensor:avgMaskF name:nil]
                              axis:0 name:nil];
    MPSGraphTensor* avgDen = [graph reductionSumWithTensor:avgMaskF axis:0 name:nil];
    MPSGraphTensor* avgVal = [graph divisionWithPrimaryTensor:avgNum
                                              secondaryTensor:[graph maximumWithPrimaryTensor:avgDen
                                                                   secondaryTensor:[graph constantWithScalar:1.0f
                                                                                        dataType:MPSDataTypeFloat32] name:nil]
                                                         name:@"avg_val"];

    // NOT EXISTS orders: gather this customer's order count from the orders subgraph result.
    MPSGraphTensor* custOrderCount = [graph gatherWithUpdatesTensor:orderCount indicesTensor:tCustkey
                                                                 axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* noOrders = [graph lessThanWithPrimaryTensor:custOrderCount secondaryTensor:halfF name:nil];

    MPSGraphTensor* inSetBool = [graph greaterThanWithPrimaryTensor:tInSet secondaryTensor:halfF name:nil];
    MPSGraphTensor* aboveAvg  = [graph greaterThanWithPrimaryTensor:tAcctbal
                                            secondaryTensor:[graph reshapeTensor:avgVal withShape:@[@1] name:nil]
                                                       name:nil];
    MPSGraphTensor* qualifyF = [graph castTensor:
                                    [graph logicalANDWithPrimaryTensor:inSetBool
                                        secondaryTensor:[graph logicalANDWithPrimaryTensor:aboveAvg
                                                                          secondaryTensor:noOrders name:nil]
                                                                       name:nil]
                                        toType:MPSDataTypeFloat32 name:@"qualify"];

    // Per-country (7 fixed groups) masked count + sum
    NSMutableArray* gCnt = [@[] mutableCopy], *gSum = [@[] mutableCopy];
    for (int g = 0; g < NUM_CODES; g++) {
        MPSGraphTensor* gConst = [graph constantWithScalar:g dataType:MPSDataTypeInt32];
        MPSGraphTensor* gMaskF = [graph castTensor:
                                       [graph equalWithPrimaryTensor:tCountryIdx secondaryTensor:gConst name:nil]
                                   toType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* combined = [graph multiplicationWithPrimaryTensor:qualifyF secondaryTensor:gMaskF name:nil];
        [gCnt addObject:[graph reductionSumWithTensor:combined axis:0 name:nil]];
        [gSum addObject:[graph reductionSumWithTensor:
                             [graph multiplicationWithPrimaryTensor:tAcctbal secondaryTensor:combined name:nil]
                         axis:0 name:nil]];
    }
    MPSGraphTensor* outCnt = [graph concatTensors:gCnt dimension:0 name:@"count"];
    MPSGraphTensor* outSum = [graph concatTensors:gSum dimension:0 name:@"sum"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffers
    // ----------------------------------------------------------------
    MPSGraphTensorData* ocTD = columnTensor(device, (void*)o_custkey.data(), M, MPSDataTypeInt32);
    MPSGraphTensorData* ckTD = columnTensor(device, (void*)c_custkey.data(), C, MPSDataTypeInt32);
    MPSGraphTensorData* abTD = columnTensor(device, (void*)c_acctbal.data(), C, MPSDataTypeFloat32);
    MPSGraphTensorData* isTD = columnTensor(device, inSet.data(),      C, MPSDataTypeFloat32);
    MPSGraphTensorData* ciTD = columnTensor(device, countryIdx.data(), C, MPSDataTypeInt32);

    NSDictionary* feeds = @{ tOCustkey: ocTD, tCustkey: ckTD, tAcctbal: abTD, tInSet: isTD, tCountryIdx: ciTD };

    id<MTLBuffer> cntBuf = allocSharedBuffer(device, NUM_CODES * sizeof(float));
    id<MTLBuffer> sumBuf = allocSharedBuffer(device, NUM_CODES * sizeof(float));
    MPSGraphTensorData* cntTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:cntBuf shape:@[@(NUM_CODES)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* sumTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:sumBuf shape:@[@(NUM_CODES)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[outCnt] = cntTD;
    results[outSum] = sumTD;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — sort by cntrycode, print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* rCnt = (float*)[cntBuf contents];
    float* rSum = (float*)[sumBuf contents];

    std::vector<int> order(NUM_CODES);
    for (int i = 0; i < NUM_CODES; i++) order[i] = i;
    std::sort(order.begin(), order.end(), [](int a, int b) {
        return std::string(COUNTRY_CODES[a]) < std::string(COUNTRY_CODES[b]);
    });

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q22 Results:\n");
    printf("+-----------+---------+--------------+\n");
    printf("| cntrycode | numcust |   totacctbal |\n");
    printf("+-----------+---------+--------------+\n");
    for (int idx : order) {
        if (rCnt[idx] < 1.0f) continue;
        printf("| %-9s | %7.0f | %12.2f |\n", COUNTRY_CODES[idx], rCnt[idx], rSum[idx]);
    }
    printf("+-----------+---------+--------------+\n");

    printf("\n  Customers: %zu  Orders: %zu\n", C, M);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q22", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
