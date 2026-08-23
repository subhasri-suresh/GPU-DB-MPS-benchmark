#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q17 — Small-Quantity-Order Revenue
//
// SELECT SUM(l_extendedprice) / 7.0 AS avg_yearly
// FROM lineitem, part
// WHERE p_partkey = l_partkey
//   AND p_brand = 'Brand#23'
//   AND p_container = 'MED BOX'
//   AND l_quantity < (
//       SELECT 0.2 * AVG(l_quantity) FROM lineitem WHERE l_partkey = p_partkey);
//
// Strategy (fully GPU except the brand/container string match):
//   CPU build: part_qualifies[partkey] = 1.0 if p_brand=='Brand#23' AND p_container=='MED BOX'.
//   MPSGraph:  the correlated subquery (per-part AVG(l_quantity) over ALL lineitem rows for
//              that part) is itself a scatter-add(sum)/scatter-add(count) pass over lineitem,
//              indexed by l_partkey — then gathered straight back per row in the SAME graph,
//              since a scatter's output tensor is just another node a later gather can consume.
//              No CPU round-trip is needed between the two passes.

void runQ17(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q17: Small-Quantity-Order Revenue ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT       },
        {3, ColType::CHAR_FIXED, 10},   // p_brand
        {6, ColType::CHAR_FIXED, 10},   // p_container
    });
    auto& p_partkey   = pCols.ints(0);
    auto& p_brand     = pCols.chars(3);
    auto& p_container = pCols.chars(6);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {1, ColType::INT  },   // l_partkey
        {4, ColType::FLOAT},   // l_quantity
        {5, ColType::FLOAT},   // l_extendedprice
    });
    auto& l_partkey  = lCols.ints(1);
    auto& l_qty      = lCols.floats(4);
    auto& l_extprice = lCols.floats(5);
    size_t N = l_partkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  Lineitem: %zu  (parse: %.1f ms)\n", p_partkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — part_qualifies[partkey] (brand + container match)
    // ----------------------------------------------------------------
    int max_partkey = *std::max_element(p_partkey.begin(), p_partkey.end());
    std::vector<float> part_qualifies((size_t)(max_partkey + 1), 0.0f);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        std::string brand = trimFixed(p_brand.data(), i, 10);
        std::string cont  = trimFixed(p_container.data(), i, 10);
        if (brand == "Brand#23" && cont == "MED BOX")
            part_qualifies[(size_t)p_partkey[i]] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tPartQual = [graph placeholderWithShape:@[@(max_partkey + 1)]
                                                    dataType:MPSDataTypeFloat32 name:@"part_qualifies"];
    MPSGraphTensor* tPartkey  = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"l_partkey"];
    MPSGraphTensor* tQty      = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeFloat32 name:@"l_qty"];
    MPSGraphTensor* tExtprice = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeFloat32 name:@"l_extprice"];

    // Correlated subquery: per-part AVG(l_quantity) over ALL lineitem rows for that part.
    MPSGraphTensor* zerosSum = [graph constantWithScalar:0.0 shape:@[@(max_partkey + 1)]
                                                dataType:MPSDataTypeFloat32];
    MPSGraphTensor* sumQtyByPart = [graph scatterWithDataTensor:zerosSum
                                                   updatesTensor:tQty
                                                   indicesTensor:tPartkey
                                                            axis:0
                                                            mode:MPSGraphScatterModeAdd
                                                            name:@"sum_qty_by_part"];
    MPSGraphTensor* onesN = [graph constantWithScalar:1.0 shape:@[@(N)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* cntByPart = [graph scatterWithDataTensor:zerosSum
                                               updatesTensor:onesN
                                               indicesTensor:tPartkey
                                                        axis:0
                                                        mode:MPSGraphScatterModeAdd
                                                        name:@"cnt_by_part"];
    MPSGraphTensor* oneF = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* safeCnt = [graph maximumWithPrimaryTensor:cntByPart secondaryTensor:oneF name:nil];
    MPSGraphTensor* avgQtyByPart = [graph divisionWithPrimaryTensor:sumQtyByPart
                                                    secondaryTensor:safeCnt name:nil];
    MPSGraphTensor* pointTwo = [graph constantWithScalar:0.2f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* thresholdByPart = [graph multiplicationWithPrimaryTensor:avgQtyByPart
                                                             secondaryTensor:pointTwo name:nil];

    // Gather per-row threshold and part-qualifies flag straight from the scatter results above.
    MPSGraphTensor* rowThreshold = [graph gatherWithUpdatesTensor:thresholdByPart
                                                     indicesTensor:tPartkey
                                                              axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* rowQualify   = [graph gatherWithUpdatesTensor:tPartQual
                                                     indicesTensor:tPartkey
                                                              axis:0 batchDimensions:0 name:nil];

    MPSGraphTensor* qtyOk   = [graph lessThanWithPrimaryTensor:tQty secondaryTensor:rowThreshold name:nil];
    MPSGraphTensor* qualOk  = [graph greaterThanWithPrimaryTensor:rowQualify
                                                   secondaryTensor:[graph constantWithScalar:0.5f
                                                                        dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:qtyOk secondaryTensor:qualOk name:nil]
                                     toType:MPSDataTypeFloat32 name:@"mask"];

    MPSGraphTensor* masked = [graph multiplicationWithPrimaryTensor:tExtprice secondaryTensor:maskF name:nil];
    MPSGraphTensor* total  = [graph reductionSumWithTensor:masked axis:0 name:@"total"];
    MPSGraphTensor* avgYearly = [graph divisionWithPrimaryTensor:total
                                                  secondaryTensor:[graph constantWithScalar:7.0f
                                                                       dataType:MPSDataTypeFloat32]
                                                             name:@"avg_yearly"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* pqTD = columnTensor(device, part_qualifies.data(),
                                            (size_t)(max_partkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* pkTD = columnTensor(device, (void*)l_partkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* qtTD = columnTensor(device, (void*)l_qty.data(),      N, MPSDataTypeFloat32);
    MPSGraphTensorData* epTD = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{ tPartQual: pqTD, tPartkey: pkTD, tQty: qtTD, tExtprice: epTD };

    id<MTLBuffer> outBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* outTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:outTD forKey:avgYearly];

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: Print + log
    // ----------------------------------------------------------------
    float avg = *(float*)[outBuf contents];
    printf("\nTPC-H Q17 Result:\n");
    printf("  avg_yearly: %.2f\n", avg);

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, 0.0);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q17", gpuMs, 0.0,
                   std::string(device.name.UTF8String), memGB);
}
