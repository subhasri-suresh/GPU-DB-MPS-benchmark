#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q20 — Potential Part Promotion
//
// SELECT s_name, s_address FROM supplier, nation
// WHERE s_suppkey IN (
//     SELECT ps_suppkey FROM partsupp
//     WHERE ps_partkey IN (SELECT p_partkey FROM part WHERE p_name LIKE 'forest%')
//       AND ps_availqty > (
//           SELECT 0.5 * SUM(l_quantity) FROM lineitem
//           WHERE l_partkey = ps_partkey AND l_suppkey = ps_suppkey
//             AND l_shipdate >= '1994-01-01' AND l_shipdate < '1995-01-01'))
//   AND s_nationkey = n_nationkey AND n_name = 'CANADA'
// ORDER BY s_name;
//
// Strategy (fully GPU except the part-name match and the final nation/print pass):
//   Same composite-key trick as Q9/Q10: TPC-H guarantees exactly 4 contiguous partsupp
//   rows per part, sorted by partkey, so row (partkey-1)*4+j is that part's j-th supplier
//   entry. For each lineitem row, resolve which of the 4 candidates matches l_suppkey and
//   scatter-add its (date-filtered) l_quantity directly into a per-partsupp-row array —
//   this *is* the "SUM(l_quantity) WHERE l_partkey=ps_partkey AND l_suppkey=ps_suppkey"
//   correlated aggregate, computed in one GPU pass with no CPU hash join. Unmatched rows
//   are diverted to a dump bucket so they can't corrupt row 0's true sum.
//   Then: per-partsupp-row availqty > 0.5*sum, AND part name matches 'forest%', is
//   scatter-OR'd (via scatter-add + threshold) into a per-supplier qualifies flag.

void runQ20(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q20: Potential Part Promotion ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT        },
        {1, ColType::CHAR_FIXED, 55},   // p_name
    });
    auto& p_partkey = pCols.ints(0);
    auto& p_name    = pCols.chars(1);

    auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl", {
        {0, ColType::INT  },   // ps_partkey
        {1, ColType::INT  },   // ps_suppkey
        {2, ColType::FLOAT},   // ps_availqty
    });
    auto& ps_partkey  = psCols.ints(0);
    auto& ps_suppkey  = psCols.ints(1);
    auto& ps_availqty = psCols.floats(2);
    size_t psN = ps_partkey.size();

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 1, ColType::INT  },   // l_partkey
        { 2, ColType::INT  },   // l_suppkey
        { 4, ColType::FLOAT},   // l_quantity
        {10, ColType::DATE },   // l_shipdate
    });
    auto& l_partkey  = lCols.ints(1);
    auto& l_suppkey  = lCols.ints(2);
    auto& l_qty      = lCols.floats(4);
    auto& l_shipdate = lCols.ints(10);
    size_t N = l_partkey.size();

    auto sup = loadColumnsMulti(g_dataset_path + "supplier.tbl", {
        {0, ColType::INT        },
        {1, ColType::CHAR_FIXED, 25}, // s_name
        {2, ColType::CHAR_FIXED, 40}, // s_address
        {3, ColType::INT        },    // s_nationkey
    });
    auto& s_suppkey   = sup.ints(0);
    auto& s_name      = sup.chars(1);
    auto& s_address   = sup.chars(2);
    auto& s_nationkey = sup.ints(3);

    auto nat = loadNation(g_dataset_path);
    int canada_key = findNationKey(nat, "CANADA");

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  PartSupp: %zu  Lineitem: %zu  Supplier: %zu  (parse: %.1f ms)\n",
           p_partkey.size(), psN, N, s_suppkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — part_forest[partkey] = 1.0 if p_name LIKE 'forest%'
    // ----------------------------------------------------------------
    int max_partkey = *std::max_element(p_partkey.begin(), p_partkey.end());
    std::vector<float> part_forest((size_t)(max_partkey + 1), 0.0f);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        if (memcmp(p_name.data() + i * 55, "forest", 6) == 0)
            part_forest[(size_t)p_partkey[i]] = 1.0f;
    }
    int max_suppkey = *std::max_element(s_suppkey.begin(), s_suppkey.end());

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tLPartkey  = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"l_partkey"];
    MPSGraphTensor* tLSuppkey  = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"l_suppkey"];
    MPSGraphTensor* tLQty      = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"l_qty"];
    MPSGraphTensor* tLShipdate = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"l_shipdate"];
    MPSGraphTensor* tPsSuppkey = [graph placeholderWithShape:@[@(psN)] dataType:MPSDataTypeInt32   name:@"ps_suppkey"];
    MPSGraphTensor* tPsPartkey = [graph placeholderWithShape:@[@(psN)] dataType:MPSDataTypeInt32   name:@"ps_partkey"];
    MPSGraphTensor* tPsAvailqty= [graph placeholderWithShape:@[@(psN)] dataType:MPSDataTypeFloat32 name:@"ps_availqty"];
    MPSGraphTensor* tPartForest= [graph placeholderWithShape:@[@(max_partkey + 1)] dataType:MPSDataTypeFloat32 name:@"part_forest"];

    // Date filter: [1994-01-01, 1995-01-01)
    MPSGraphTensor* startDate = [graph constantWithScalar:19940101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* endDate   = [graph constantWithScalar:19950101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateOkF   = [graph castTensor:
                                    [graph logicalANDWithPrimaryTensor:
                                        [graph greaterThanOrEqualToWithPrimaryTensor:tLShipdate
                                                                     secondaryTensor:startDate name:nil]
                                    secondaryTensor:
                                        [graph lessThanWithPrimaryTensor:tLShipdate
                                                         secondaryTensor:endDate name:nil]
                                    name:nil]
                                     toType:MPSDataTypeFloat32 name:nil];

    // Resolve (l_partkey,l_suppkey) -> partsupp row index, exploiting the fixed 4-rows-per-part layout.
    MPSGraphTensor* oneI  = [graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* fourI = [graph constantWithScalar:4 dataType:MPSDataTypeInt32];
    MPSGraphTensor* psBase = [graph multiplicationWithPrimaryTensor:
                                  [graph subtractionWithPrimaryTensor:tLPartkey secondaryTensor:oneI name:nil]
                              secondaryTensor:fourI name:@"ps_base"];

    MPSGraphTensor* idxSum  = [graph constantWithScalar:0.0 shape:@[@(N)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* anyMatch= [graph constantWithScalar:0.0 shape:@[@(N)] dataType:MPSDataTypeFloat32];
    for (int j = 0; j < 4; j++) {
        MPSGraphTensor* jI  = [graph constantWithScalar:j dataType:MPSDataTypeInt32];
        MPSGraphTensor* idx = [graph additionWithPrimaryTensor:psBase secondaryTensor:jI name:nil];
        MPSGraphTensor* candSuppkey = [graph gatherWithUpdatesTensor:tPsSuppkey indicesTensor:idx
                                                                  axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* matchF = [graph castTensor:
                                      [graph equalWithPrimaryTensor:candSuppkey secondaryTensor:tLSuppkey name:nil]
                                  toType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* idxF = [graph castTensor:idx toType:MPSDataTypeFloat32 name:nil];
        idxSum   = [graph additionWithPrimaryTensor:idxSum
                              secondaryTensor:[graph multiplicationWithPrimaryTensor:matchF secondaryTensor:idxF name:nil]
                              name:nil];
        anyMatch = [graph additionWithPrimaryTensor:anyMatch secondaryTensor:matchF name:nil];
    }
    // Unmatched rows (anyMatch==0) are diverted to a dump bucket at index psN.
    MPSGraphTensor* anyMatchBool = [graph greaterThanWithPrimaryTensor:anyMatch
                                                        secondaryTensor:[graph constantWithScalar:0.5f
                                                                             dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* dumpOffset = [graph selectWithPredicateTensor:anyMatchBool
                                                truePredicateTensor:[graph constantWithScalar:0.0 shape:@[@(N)] dataType:MPSDataTypeFloat32]
                                               falsePredicateTensor:[graph constantWithScalar:(double)psN shape:@[@(N)] dataType:MPSDataTypeFloat32]
                                                               name:nil];
    MPSGraphTensor* resolvedIdxF = [graph additionWithPrimaryTensor:idxSum secondaryTensor:dumpOffset name:nil];
    MPSGraphTensor* resolvedIdx  = [graph castTensor:resolvedIdxF toType:MPSDataTypeInt32 name:@"resolved_idx"];

    // Per-partsupp-row SUM(l_quantity) for 1994, via scatter-add into a (psN+1)-sized array
    // (the extra slot is the dump bucket for unmatched lineitem rows).
    MPSGraphTensor* dateMaskedQty = [graph multiplicationWithPrimaryTensor:tLQty secondaryTensor:dateOkF name:nil];
    MPSGraphTensor* zerosPS1 = [graph constantWithScalar:0.0 shape:@[@(psN + 1)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* sumQtyPS = [graph scatterWithDataTensor:zerosPS1
                                              updatesTensor:dateMaskedQty
                                              indicesTensor:resolvedIdx
                                                       axis:0
                                                       mode:MPSGraphScatterModeAdd
                                                       name:@"sum_qty_ps"];
    MPSGraphTensor* sumQtyPSValid = [graph sliceTensor:sumQtyPS dimension:0 start:0 length:(NSInteger)psN name:@"sum_qty_ps_valid"];

    // Per-partsupp-row qualification: availqty > 0.5*sum AND part is 'forest%'.
    MPSGraphTensor* halfSum = [graph multiplicationWithPrimaryTensor:sumQtyPSValid
                                                     secondaryTensor:[graph constantWithScalar:0.5f
                                                                          dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* availOk = [graph greaterThanWithPrimaryTensor:tPsAvailqty secondaryTensor:halfSum name:nil];
    MPSGraphTensor* forestMatch = [graph gatherWithUpdatesTensor:tPartForest indicesTensor:tPsPartkey
                                                              axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* forestOk = [graph greaterThanWithPrimaryTensor:forestMatch
                                                    secondaryTensor:[graph constantWithScalar:0.5f
                                                                         dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* qualifyF = [graph castTensor:
                                    [graph logicalANDWithPrimaryTensor:availOk secondaryTensor:forestOk name:nil]
                                        toType:MPSDataTypeFloat32 name:@"qualify"];

    // Per-supplier qualifies flag: OR (via scatter-add + threshold) over all its partsupp rows.
    MPSGraphTensor* zerosSupp = [graph constantWithScalar:0.0 shape:@[@(max_suppkey + 1)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* supplierQualifyCount = [graph scatterWithDataTensor:zerosSupp
                                                          updatesTensor:qualifyF
                                                          indicesTensor:tPsSuppkey
                                                                   axis:0
                                                                   mode:MPSGraphScatterModeAdd
                                                                   name:@"supplier_qualify_count"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* lpkTD = columnTensor(device, (void*)l_partkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* lskTD = columnTensor(device, (void*)l_suppkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* lqTD  = columnTensor(device, (void*)l_qty.data(),      N, MPSDataTypeFloat32);
    MPSGraphTensorData* lsdTD = columnTensor(device, (void*)l_shipdate.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* pskTD = columnTensor(device, (void*)ps_suppkey.data(),  psN, MPSDataTypeInt32);
    MPSGraphTensorData* ppkTD = columnTensor(device, (void*)ps_partkey.data(),  psN, MPSDataTypeInt32);
    MPSGraphTensorData* paqTD = columnTensor(device, (void*)ps_availqty.data(), psN, MPSDataTypeFloat32);
    MPSGraphTensorData* pfTD  = columnTensor(device, part_forest.data(), (size_t)(max_partkey + 1), MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tLPartkey: lpkTD, tLSuppkey: lskTD, tLQty: lqTD, tLShipdate: lsdTD,
        tPsSuppkey: pskTD, tPsPartkey: ppkTD, tPsAvailqty: paqTD, tPartForest: pfTD
    };

    id<MTLBuffer> suppQualBuf = allocSharedBuffer(device, (size_t)(max_suppkey + 1) * sizeof(float));
    MPSGraphTensorData* suppQualTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:suppQualBuf shape:@[@(max_suppkey + 1)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[supplierQualifyCount] = suppQualTD;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — filter to Canada, sort by name, print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* suppQual = (float*)[suppQualBuf contents];

    struct Row { std::string name, address; };
    std::vector<Row> rows;
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        int sk = s_suppkey[i];
        if (s_nationkey[i] != canada_key) continue;
        if (sk < 0 || sk > max_suppkey || suppQual[sk] < 0.5f) continue;
        rows.push_back({trimFixed(s_name.data(), i, 25), trimFixed(s_address.data(), i, 40)});
    }
    std::sort(rows.begin(), rows.end(), [](const Row& a, const Row& b) { return a.name < b.name; });

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q20 Results (first 15 of %zu):\n", rows.size());
    printf("+--------------------+------------------------------------------+\n");
    printf("| s_name             | s_address                                |\n");
    printf("+--------------------+------------------------------------------+\n");
    size_t show = std::min(rows.size(), (size_t)15);
    for (size_t i = 0; i < show; i++)
        printf("| %-18s | %-40s |\n", rows[i].name.c_str(), rows[i].address.c_str());
    printf("+--------------------+------------------------------------------+\n");

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q20", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
