#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q19 — Discounted Revenue
//
// SELECT SUM(l_extendedprice * (1 - l_discount)) AS revenue
// FROM lineitem, part
// WHERE ( p_partkey = l_partkey AND p_brand = 'Brand#12'
//         AND p_container IN ('SM CASE','SM BOX','SM PACK','SM PKG')
//         AND l_quantity BETWEEN 1  AND 11  AND p_size BETWEEN 1 AND 5
//         AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON' )
//    OR ( p_partkey = l_partkey AND p_brand = 'Brand#23'
//         AND p_container IN ('MED BAG','MED BOX','MED PKG','MED PACK')
//         AND l_quantity BETWEEN 10 AND 20  AND p_size BETWEEN 1 AND 10
//         AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON' )
//    OR ( p_partkey = l_partkey AND p_brand = 'Brand#34'
//         AND p_container IN ('LG CASE','LG BOX','LG PACK','LG PKG')
//         AND l_quantity BETWEEN 20 AND 30  AND p_size BETWEEN 1 AND 15
//         AND l_shipmode IN ('AIR','AIR REG') AND l_shipinstruct = 'DELIVER IN PERSON' );
//
// Strategy (fully GPU except the string-valued part/shipping columns):
//   CPU build: 3 per-part group flags (brand+container+size all static per part, so each
//              is precomputed once as a dense array indexed by partkey), and a per-row
//              ship_ok flag (shipmode/shipinstruct are per-lineitem-row strings).
//   MPSGraph:  gather the 3 group flags per row, AND each with its own quantity range,
//              OR the three together, AND with ship_ok, sum masked revenue.

static bool containerInSet(const std::string& c, std::initializer_list<const char*> set) {
    for (auto s : set) if (c == s) return true;
    return false;
}

void runQ19(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q19: Discounted Revenue ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT        },
        {3, ColType::CHAR_FIXED, 10},   // p_brand
        {5, ColType::INT        },      // p_size
        {6, ColType::CHAR_FIXED, 10},   // p_container
    });
    auto& p_partkey   = pCols.ints(0);
    auto& p_brand     = pCols.chars(3);
    auto& p_size      = pCols.ints(5);
    auto& p_container = pCols.chars(6);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 1, ColType::INT        },   // l_partkey
        { 4, ColType::FLOAT      },   // l_quantity
        { 5, ColType::FLOAT      },   // l_extendedprice
        { 6, ColType::FLOAT      },   // l_discount
        {13, ColType::CHAR_FIXED, 25}, // l_shipinstruct
        {14, ColType::CHAR_FIXED, 10}, // l_shipmode
    });
    auto& l_partkey     = lCols.ints(1);
    auto& l_qty         = lCols.floats(4);
    auto& l_extprice    = lCols.floats(5);
    auto& l_discount    = lCols.floats(6);
    auto& l_shipinstruct = lCols.chars(13);
    auto& l_shipmode    = lCols.chars(14);
    size_t N = l_partkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  Lineitem: %zu  (parse: %.1f ms)\n", p_partkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — per-part group flags (brand + container-set + size range)
    // ----------------------------------------------------------------
    int max_partkey = *std::max_element(p_partkey.begin(), p_partkey.end());
    std::vector<float> group1((size_t)(max_partkey + 1), 0.0f);
    std::vector<float> group2((size_t)(max_partkey + 1), 0.0f);
    std::vector<float> group3((size_t)(max_partkey + 1), 0.0f);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        std::string brand = trimFixed(p_brand.data(), i, 10);
        std::string cont  = trimFixed(p_container.data(), i, 10);
        int sz = p_size[i];
        int pk = p_partkey[i];
        if (brand == "Brand#12" && sz >= 1 && sz <= 5 &&
            containerInSet(cont, {"SM CASE", "SM BOX", "SM PACK", "SM PKG"}))
            group1[(size_t)pk] = 1.0f;
        if (brand == "Brand#23" && sz >= 1 && sz <= 10 &&
            containerInSet(cont, {"MED BAG", "MED BOX", "MED PKG", "MED PACK"}))
            group2[(size_t)pk] = 1.0f;
        if (brand == "Brand#34" && sz >= 1 && sz <= 15 &&
            containerInSet(cont, {"LG CASE", "LG BOX", "LG PACK", "LG PKG"}))
            group3[(size_t)pk] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — per-row ship_ok flag (shipmode in {AIR,AIR REG} AND shipinstruct == DELIVER IN PERSON)
    // ----------------------------------------------------------------
    std::vector<float> ship_ok(N);
    for (size_t i = 0; i < N; i++) {
        std::string mode = trimFixed(l_shipmode.data(), i, 10);
        std::string inst = trimFixed(l_shipinstruct.data(), i, 25);
        bool ok = (mode == "AIR" || mode == "AIR REG") && inst == "DELIVER IN PERSON";
        ship_ok[i] = ok ? 1.0f : 0.0f;
    }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tGroup1  = [graph placeholderWithShape:@[@(max_partkey + 1)] dataType:MPSDataTypeFloat32 name:@"group1"];
    MPSGraphTensor* tGroup2  = [graph placeholderWithShape:@[@(max_partkey + 1)] dataType:MPSDataTypeFloat32 name:@"group2"];
    MPSGraphTensor* tGroup3  = [graph placeholderWithShape:@[@(max_partkey + 1)] dataType:MPSDataTypeFloat32 name:@"group3"];
    MPSGraphTensor* tPartkey = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"l_partkey"];
    MPSGraphTensor* tQty     = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"l_qty"];
    MPSGraphTensor* tPrice   = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"l_extprice"];
    MPSGraphTensor* tDiscount= [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"l_discount"];
    MPSGraphTensor* tShipOk  = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"ship_ok"];

    MPSGraphTensor* g1Match = [graph gatherWithUpdatesTensor:tGroup1 indicesTensor:tPartkey axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* g2Match = [graph gatherWithUpdatesTensor:tGroup2 indicesTensor:tPartkey axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* g3Match = [graph gatherWithUpdatesTensor:tGroup3 indicesTensor:tPartkey axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* halfF = [graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32];

    auto qtyRange = [&](float lo, float hi) {
        MPSGraphTensor* loT = [graph constantWithScalar:lo dataType:MPSDataTypeFloat32];
        MPSGraphTensor* hiT = [graph constantWithScalar:hi dataType:MPSDataTypeFloat32];
        return [graph logicalANDWithPrimaryTensor:
                    [graph greaterThanOrEqualToWithPrimaryTensor:tQty secondaryTensor:loT name:nil]
                secondaryTensor:
                    [graph lessThanOrEqualToWithPrimaryTensor:tQty secondaryTensor:hiT name:nil] name:nil];
    };

    MPSGraphTensor* match1 = [graph logicalANDWithPrimaryTensor:
                                  [graph greaterThanWithPrimaryTensor:g1Match secondaryTensor:halfF name:nil]
                              secondaryTensor:qtyRange(1.0f, 11.0f) name:nil];
    MPSGraphTensor* match2 = [graph logicalANDWithPrimaryTensor:
                                  [graph greaterThanWithPrimaryTensor:g2Match secondaryTensor:halfF name:nil]
                              secondaryTensor:qtyRange(10.0f, 20.0f) name:nil];
    MPSGraphTensor* match3 = [graph logicalANDWithPrimaryTensor:
                                  [graph greaterThanWithPrimaryTensor:g3Match secondaryTensor:halfF name:nil]
                              secondaryTensor:qtyRange(20.0f, 30.0f) name:nil];

    MPSGraphTensor* anyMatch = [graph logicalORWithPrimaryTensor:match1
                                              secondaryTensor:[graph logicalORWithPrimaryTensor:match2
                                                                             secondaryTensor:match3 name:nil]
                                                              name:nil];
    MPSGraphTensor* shipOkBool = [graph greaterThanWithPrimaryTensor:tShipOk secondaryTensor:halfF name:nil];
    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:anyMatch secondaryTensor:shipOkBool name:nil]
                                     toType:MPSDataTypeFloat32 name:@"mask"];

    MPSGraphTensor* oneF = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* discPrice = [graph multiplicationWithPrimaryTensor:tPrice
                                     secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                          secondaryTensor:tDiscount name:nil]
                                     name:nil];
    MPSGraphTensor* masked  = [graph multiplicationWithPrimaryTensor:discPrice secondaryTensor:maskF name:nil];
    MPSGraphTensor* revenue = [graph reductionSumWithTensor:masked axis:0 name:@"revenue"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* g1TD = columnTensor(device, group1.data(), (size_t)(max_partkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* g2TD = columnTensor(device, group2.data(), (size_t)(max_partkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* g3TD = columnTensor(device, group3.data(), (size_t)(max_partkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* pkTD = columnTensor(device, (void*)l_partkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* qtTD = columnTensor(device, (void*)l_qty.data(),      N, MPSDataTypeFloat32);
    MPSGraphTensorData* prTD = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* soTD = columnTensor(device, ship_ok.data(),           N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tGroup1: g1TD, tGroup2: g2TD, tGroup3: g3TD,
        tPartkey: pkTD, tQty: qtTD, tPrice: prTD, tDiscount: diTD, tShipOk: soTD
    };

    id<MTLBuffer> outBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* outTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:outTD forKey:revenue];

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: Print + log
    // ----------------------------------------------------------------
    float rev = *(float*)[outBuf contents];
    printf("\nTPC-H Q19 Result:\n");
    printf("  revenue: $%.2f\n", rev);

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, 0.0);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q19", gpuMs, 0.0,
                   std::string(device.name.UTF8String), memGB);
}
