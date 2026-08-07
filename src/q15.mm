#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q15 — Top Supplier
//
// revenue0(supplier_no, total_revenue) AS
//   SELECT l_suppkey, SUM(l_extendedprice*(1-l_discount))
//   FROM lineitem WHERE l_shipdate >= '1996-01-01' AND l_shipdate < '1996-04-01'
//   GROUP BY l_suppkey
//
// SELECT s_suppkey, s_name, s_address, s_phone, total_revenue
// FROM supplier, revenue0
// WHERE s_suppkey = supplier_no AND total_revenue = (SELECT MAX(total_revenue) FROM revenue0)
// ORDER BY s_suppkey
//
// Strategy: pure MPSGraph, no CPU-side joins needed.
//   MPSGraph:  mask lineitem rows by ship-date window, scatter masked revenue into a
//              dense (max_suppkey+1) per-supplier array (the GROUP BY).
//   CPU post:  scan for the max value, print every supplier tied for that max.

void runQ15(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q15: Top Supplier ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {2, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE},
    });
    auto& l_suppkey  = lCols.ints(2);
    auto& l_extprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto& l_shipdate = lCols.ints(10);
    size_t N = l_suppkey.size();

    auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 25}, {2, ColType::CHAR_FIXED, 40},
        {4, ColType::CHAR_FIXED, 15},
    });
    auto& s_suppkey = sCols.ints(0);
    auto& s_name    = sCols.chars(1);
    auto& s_address = sCols.chars(2);
    auto& s_phone   = sCols.chars(4);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Lineitem: %zu  Supplier: %zu  (parse: %.1f ms)\n", N, s_suppkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: Build MPSGraph
    // ----------------------------------------------------------------
    int max_suppkey = *std::max_element(s_suppkey.begin(), s_suppkey.end());

    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tSuppkey  = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeInt32   name:@"l_suppkey"];
    MPSGraphTensor* tShipdate = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeInt32   name:@"shipdate"];
    MPSGraphTensor* tExtprice = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeFloat32 name:@"discount"];

    MPSGraphTensor* dateStart = [graph constantWithScalar:19960101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateEnd   = [graph constantWithScalar:19960401 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateOk = [graph logicalANDWithPrimaryTensor:
                                  [graph greaterThanOrEqualToWithPrimaryTensor:tShipdate
                                                               secondaryTensor:dateStart name:nil]
                              secondaryTensor:
                                  [graph lessThanWithPrimaryTensor:tShipdate
                                                   secondaryTensor:dateEnd name:nil]
                              name:nil];
    MPSGraphTensor* maskF = [graph castTensor:dateOk toType:MPSDataTypeFloat32 name:@"mask"];

    // revenue = extprice * (1 - discount) * mask
    MPSGraphTensor* oneF    = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue = [graph multiplicationWithPrimaryTensor:
                                   [graph multiplicationWithPrimaryTensor:tExtprice
                                       secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                           secondaryTensor:tDiscount name:nil]
                                       name:nil]
                               secondaryTensor:maskF name:@"revenue"];

    // GROUP BY l_suppkey: scatter into a dense per-supplier revenue array
    MPSGraphTensor* zerosRevMap = [graph constantWithScalar:0.0 shape:@[@(max_suppkey + 1)]
                                                   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenueMap = [graph scatterWithDataTensor:zerosRevMap
                                               updatesTensor:revenue
                                               indicesTensor:tSuppkey
                                                        axis:0
                                                        mode:MPSGraphScatterModeAdd
                                                        name:@"revenue_map"];

    // ----------------------------------------------------------------
    // Step 3: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* skTD = columnTensor(device, (void*)l_suppkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* sdTD = columnTensor(device, (void*)l_shipdate.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* epTD = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tSuppkey: skTD, tShipdate: sdTD, tExtprice: epTD, tDiscount: diTD,
    };

    id<MTLBuffer> revMapBuf = allocSharedBuffer(device, (size_t)(max_suppkey + 1) * sizeof(float));
    MPSGraphTensorData* revMapTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:revMapBuf shape:@[@(max_suppkey + 1)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[revenueMap] = revMapTD;

    // ----------------------------------------------------------------
    // Step 4: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 5: CPU post — find max revenue, print matching supplier(s)
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* revenue_map = (float*)[revMapBuf contents];
    float max_revenue = 0.0f;
    for (int i = 0; i <= max_suppkey; i++)
        if (revenue_map[i] > max_revenue) max_revenue = revenue_map[i];

    std::vector<size_t> supp_index((size_t)(max_suppkey + 1), SIZE_MAX);
    for (size_t i = 0; i < s_suppkey.size(); i++)
        supp_index[(size_t)s_suppkey[i]] = i;

    printf("\nTPC-H Q15 Results:\n");
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    printf("| suppkey |           s_name |        s_address |          s_phone |    total_revenue |\n");
    printf("+---------+------------------+------------------+------------------+------------------+\n");
    for (int i = 0; i <= max_suppkey; i++) {
        if (revenue_map[i] == max_revenue && max_revenue > 0.0f) {
            size_t si = supp_index[(size_t)i];
            if (si == SIZE_MAX) continue;
            printf("| %7d | %-16s | %-16s | %-16s | %16.2f |\n",
                   i,
                   trimFixed(s_name.data(), si, 25).c_str(),
                   trimFixed(s_address.data(), si, 40).c_str(),
                   trimFixed(s_phone.data(), si, 15).c_str(),
                   revenue_map[i]);
        }
    }
    printf("+---------+------------------+------------------+------------------+------------------+\n");

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q15", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
