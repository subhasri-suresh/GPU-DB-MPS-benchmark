#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q11 — Important Stock Identification
//
// SELECT ps_partkey, SUM(ps_supplycost * ps_availqty) AS value
// FROM partsupp, supplier, nation
// WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY'
// GROUP BY ps_partkey
// HAVING SUM(ps_supplycost * ps_availqty) >
//        (SELECT SUM(ps_supplycost * ps_availqty) * 0.0001
//         FROM partsupp, supplier, nation
//         WHERE ps_suppkey = s_suppkey AND s_nationkey = n_nationkey AND n_name = 'GERMANY')
// ORDER BY value DESC
//
// Strategy (mirrors the raw-Metal baseline's single-pass aggregate + partial-sum kernel):
//   CPU build: supp_germany_mask[suppkey] = 1.0 if supplier is in GERMANY, else 0.0.
//   MPSGraph:  gather the mask per partsupp row, compute masked value = supplycost *
//              availqty * mask; scatter into a dense (max_partkey+1) value array (the
//              GROUP BY), and in the same graph reductionSum the masked values into a
//              scalar (the correlated subquery's global sum) — both read from one pass
//              over partsupp, avoiding the raw kernel's separate partial-sum reduction.
//   CPU post:  threshold = global_sum * 0.0001; keep partkeys whose value exceeds it,
//              sort descending, print top 20.

void runQ11(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q11: Important Stock Identification ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto nat = loadNation(g_dataset_path);

    auto sup = loadSupplierBasic(g_dataset_path);
    auto& s_suppkey   = sup.suppkey;
    auto& s_nationkey = sup.nationkey;

    auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl", {
        {0, ColType::INT}, {1, ColType::INT}, {2, ColType::FLOAT}, {3, ColType::FLOAT},
    });
    auto& ps_partkey    = psCols.ints(0);
    auto& ps_suppkey    = psCols.ints(1);
    auto& ps_availqty   = psCols.floats(2);
    auto& ps_supplycost = psCols.floats(3);
    size_t M = ps_partkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Supplier: %zu  PartSupp: %zu  (parse: %.1f ms)\n", s_suppkey.size(), M, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — GERMANY nationkey, supplier mask
    // ----------------------------------------------------------------
    int germany_nk = findNationKey(nat, "GERMANY");
    if (germany_nk == -1) {
        fprintf(stderr, "  GERMANY not found\n");
        return;
    }

    int max_suppkey = *std::max_element(s_suppkey.begin(), s_suppkey.end());
    int max_partkey = *std::max_element(ps_partkey.begin(), ps_partkey.end());

    std::vector<float> supp_germany_mask((size_t)(max_suppkey + 1), 0.0f);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        if (s_nationkey[i] == germany_nk)
            supp_germany_mask[(size_t)s_suppkey[i]] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tSuppMask = [graph placeholderWithShape:@[@(max_suppkey + 1)]
                                                   dataType:MPSDataTypeFloat32 name:@"supp_mask"];
    MPSGraphTensor* tPsPartkey = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeInt32   name:@"ps_partkey"];
    MPSGraphTensor* tPsSuppkey = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeInt32   name:@"ps_suppkey"];
    MPSGraphTensor* tSupplyCost = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeFloat32 name:@"supplycost"];
    MPSGraphTensor* tAvailQty  = [graph placeholderWithShape:@[@(M)]
                                                    dataType:MPSDataTypeFloat32 name:@"availqty"];

    // Gather GERMANY supplier mask per partsupp row
    MPSGraphTensor* maskF = [graph gatherWithUpdatesTensor:tSuppMask
                                             indicesTensor:tPsSuppkey
                                                      axis:0 batchDimensions:0 name:@"mask"];

    // value = supplycost * availqty * mask
    MPSGraphTensor* value = [graph multiplicationWithPrimaryTensor:
                                 [graph multiplicationWithPrimaryTensor:tSupplyCost
                                     secondaryTensor:tAvailQty name:nil]
                             secondaryTensor:maskF name:@"value"];

    // GROUP BY ps_partkey: scatter into a dense value array
    MPSGraphTensor* zerosValueMap = [graph constantWithScalar:0.0 shape:@[@(max_partkey + 1)]
                                                     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* valueMap = [graph scatterWithDataTensor:zerosValueMap
                                              updatesTensor:value
                                              indicesTensor:tPsPartkey
                                                       axis:0
                                                       mode:MPSGraphScatterModeAdd
                                                       name:@"value_map"];

    // Correlated subquery: global sum of the same masked values
    MPSGraphTensor* globalSum = [graph reductionSumWithTensor:value axis:0 name:@"global_sum"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffers
    // ----------------------------------------------------------------
    MPSGraphTensorData* smTD = columnTensor(device, supp_germany_mask.data(),
                                            (size_t)(max_suppkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* pkTD = columnTensor(device, (void*)ps_partkey.data(),    M, MPSDataTypeInt32);
    MPSGraphTensorData* skTD = columnTensor(device, (void*)ps_suppkey.data(),    M, MPSDataTypeInt32);
    MPSGraphTensorData* scTD = columnTensor(device, (void*)ps_supplycost.data(), M, MPSDataTypeFloat32);
    MPSGraphTensorData* aqTD = columnTensor(device, (void*)ps_availqty.data(),   M, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tSuppMask: smTD, tPsPartkey: pkTD, tPsSuppkey: skTD,
        tSupplyCost: scTD, tAvailQty: aqTD,
    };

    id<MTLBuffer> valueMapBuf = allocSharedBuffer(device, (size_t)(max_partkey + 1) * sizeof(float));
    id<MTLBuffer> globalSumBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* valueMapTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:valueMapBuf shape:@[@(max_partkey + 1)] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* globalSumTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:globalSumBuf shape:@[@1] dataType:MPSDataTypeFloat32];

    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[valueMap]  = valueMapTD;
    results[globalSum] = globalSumTD;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — threshold filter, sort, print top 20
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    double global_sum = (double)(*(float*)[globalSumBuf contents]);
    double threshold = global_sum * 0.0001;

    float* value_map = (float*)[valueMapBuf contents];
    struct Q11Result { int partkey; double value; };
    std::vector<Q11Result> qres;
    for (int i = 0; i <= max_partkey; i++) {
        if (value_map[i] > threshold)
            qres.push_back({i, (double)value_map[i]});
    }
    std::sort(qres.begin(), qres.end(), [](const Q11Result& a, const Q11Result& b) {
        return a.value > b.value;
    });

    printf("\nTPC-H Query 11 Results (Top 20 of %zu):\n", qres.size());
    printf("+-----------+------------------+\n");
    printf("| ps_partkey|            value |\n");
    printf("+-----------+------------------+\n");
    size_t limit = std::min(qres.size(), (size_t)20);
    for (size_t i = 0; i < limit; i++)
        printf("| %9d | %16.2f |\n", qres[i].partkey, qres[i].value);
    printf("+-----------+------------------+\n");
    printf("Total qualifying rows: %zu, Global sum: %.2f, Threshold: %.2f\n",
           qres.size(), global_sum, threshold);

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", M);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q11", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
