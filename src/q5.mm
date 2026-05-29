#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q5 — Local Supplier Volume
//
// SELECT n_name, SUM(l_extendedprice*(1-l_discount)) AS revenue
// FROM customer, orders, lineitem, supplier, nation, region
// WHERE c_custkey    = o_custkey
//   AND l_orderkey   = o_orderkey
//   AND l_suppkey    = s_suppkey
//   AND c_nationkey  = s_nationkey
//   AND s_nationkey  = n_nationkey
//   AND n_regionkey  = r_regionkey
//   AND r_name       = 'ASIA'
//   AND o_orderdate >= '1994-01-01'
//   AND o_orderdate  < '1995-01-01'
// GROUP BY n_name ORDER BY revenue DESC
//
// Hybrid strategy:
//   CPU build: ASIA nation set, customer-nation map, supplier-nation map,
//              orders-nation map (orderkey → nationkey filtered by date + ASIA customer).
//   MPSGraph:  gather order/supp nations per lineitem; mask on match; scatter revenue
//              into 25-element nation array.
//   CPU post:  reuse postProcessQ5 for sorted table.

void runQ5(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q5: Local Supplier Volume ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT}, {3, ColType::INT},   // c_custkey, c_nationkey
    });
    auto& c_custkey  = cCols.ints(0);
    auto& c_nationkey = cCols.ints(3);

    auto sup = loadSupplierBasic(g_dataset_path);
    auto& s_suppkey   = sup.suppkey;
    auto& s_nationkey = sup.nationkey;

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT }, {1, ColType::INT }, {4, ColType::DATE},
    });
    auto& o_orderkey  = oCols.ints(0);
    auto& o_custkey   = oCols.ints(1);
    auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {0, ColType::INT  }, {2, ColType::INT  },
        {5, ColType::FLOAT}, {6, ColType::FLOAT},
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_suppkey  = lCols.ints(2);
    auto& l_extprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    size_t N = l_orderkey.size();

    auto nat = loadNation(g_dataset_path, /*with_regionkey=*/true);
    auto reg = loadRegion(g_dataset_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Customer: %zu  Supplier: %zu  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n",
           c_custkey.size(), s_suppkey.size(), o_orderkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — identify ASIA nations
    // ----------------------------------------------------------------
    int asia_rk = findRegionKey(reg.regionkey, reg.name.data(), RegionData::NAME_WIDTH, "ASIA");
    if (asia_rk == -1) { fprintf(stderr, "  ASIA region not found\n"); return; }

    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);

    // Set of nationkeys that are in ASIA
    std::vector<bool> is_asia_nation(26, false);  // nationkeys are 0..24
    for (size_t i = 0; i < nat.nationkey.size(); i++) {
        if (nat.regionkey[i] == asia_rk && nat.nationkey[i] < 25)
            is_asia_nation[(size_t)nat.nationkey[i]] = true;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — build lookup maps
    // ----------------------------------------------------------------
    int max_custkey  = *std::max_element(c_custkey.begin(),  c_custkey.end());
    int max_suppkey  = *std::max_element(s_suppkey.begin(),  s_suppkey.end());
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());

    // cust_nation_map[custkey] = nationkey if ASIA, else -1
    std::vector<int32_t> cust_nation_map((size_t)(max_custkey + 1), -1);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        int nk = c_nationkey[i];
        if (nk >= 0 && nk < 25 && is_asia_nation[(size_t)nk])
            cust_nation_map[(size_t)c_custkey[i]] = nk;
    }

    // supp_nation_map[suppkey] = nationkey if ASIA, else -1
    std::vector<int32_t> supp_nation_map((size_t)(max_suppkey + 1), -1);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        int nk = s_nationkey[i];
        if (nk >= 0 && nk < 25 && is_asia_nation[(size_t)nk])
            supp_nation_map[(size_t)s_suppkey[i]] = nk;
    }

    // orders_nation_map[orderkey] = nationkey if date in range AND ASIA customer, else -1
    const int DATE_START = 19940101, DATE_END = 19950101;
    std::vector<int32_t> orders_nation_map((size_t)(max_orderkey + 1), -1);
    for (size_t i = 0; i < o_orderkey.size(); i++) {
        if (o_orderdate[i] < DATE_START || o_orderdate[i] >= DATE_END) continue;
        int ck = o_custkey[i];
        if (ck > max_custkey || cust_nation_map[(size_t)ck] < 0) continue;
        orders_nation_map[(size_t)o_orderkey[i]] = cust_nation_map[(size_t)ck];
    }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tOrdNatMap  = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                     dataType:MPSDataTypeInt32   name:@"ord_nat_map"];
    MPSGraphTensor* tSuppNatMap = [graph placeholderWithShape:@[@(max_suppkey + 1)]
                                                     dataType:MPSDataTypeInt32   name:@"supp_nat_map"];
    MPSGraphTensor* tLOrderkey  = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tLSuppkey   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeInt32   name:@"l_suppkey"];
    MPSGraphTensor* tExtprice   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeFloat32 name:@"discount"];

    // Gather per-lineitem nationkeys from both maps
    MPSGraphTensor* ordNation  = [graph gatherWithUpdatesTensor:tOrdNatMap
                                                  indicesTensor:tLOrderkey
                                                           axis:0 batchDimensions:0 name:@"ord_nat"];
    MPSGraphTensor* suppNation = [graph gatherWithUpdatesTensor:tSuppNatMap
                                                  indicesTensor:tLSuppkey
                                                           axis:0 batchDimensions:0 name:@"supp_nat"];

    // Validity and match conditions (int32 comparisons → bool)
    MPSGraphTensor* zeroI      = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* validOrd   = [graph greaterThanOrEqualToWithPrimaryTensor:ordNation
                                                              secondaryTensor:zeroI name:nil];
    MPSGraphTensor* validSupp  = [graph greaterThanOrEqualToWithPrimaryTensor:suppNation
                                                               secondaryTensor:zeroI name:nil];
    MPSGraphTensor* natMatch   = [graph equalWithPrimaryTensor:ordNation
                                               secondaryTensor:suppNation name:nil];

    // Combined mask: all three conditions → float
    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:
                                     [graph logicalANDWithPrimaryTensor:validOrd
                                                        secondaryTensor:validSupp name:nil]
                                  secondaryTensor:natMatch name:nil]
                             toType:MPSDataTypeFloat32 name:@"mask"];

    // Revenue = extprice * (1 - discount) * mask
    MPSGraphTensor* oneF    = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue = [graph multiplicationWithPrimaryTensor:
                                   [graph multiplicationWithPrimaryTensor:tExtprice
                                       secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                           secondaryTensor:tDiscount name:nil]
                                       name:nil]
                               secondaryTensor:maskF name:@"revenue"];

    // Clamp order nation to ≥ 0 for safe scatter index (invalid rows carry 0 revenue)
    MPSGraphTensor* clampedNat = [graph maximumWithPrimaryTensor:ordNation
                                                secondaryTensor:zeroI name:@"clamped_nat"];

    // Scatter into 25-element nation revenue array
    MPSGraphTensor* zeros25       = [graph constantWithScalar:0.0 shape:@[@25]
                                                     dataType:MPSDataTypeFloat32];
    MPSGraphTensor* nationRevenue = [graph scatterWithDataTensor:zeros25
                                                  updatesTensor:revenue
                                                  indicesTensor:clampedNat
                                                           axis:0
                                                           mode:MPSGraphScatterModeAdd
                                                           name:@"nation_revenue"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* onmTD  = columnTensor(device, orders_nation_map.data(),
                                              (size_t)(max_orderkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* snmTD  = columnTensor(device, supp_nation_map.data(),
                                              (size_t)(max_suppkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* okTD   = columnTensor(device, (void*)l_orderkey.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* skTD   = columnTensor(device, (void*)l_suppkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* epTD   = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD   = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tOrdNatMap: onmTD, tSuppNatMap: snmTD,
        tLOrderkey: okTD,  tLSuppkey:   skTD,
        tExtprice:  epTD,  tDiscount:   diTD,
    };

    id<MTLBuffer> revBuf = allocSharedBuffer(device, 25 * sizeof(float));
    MPSGraphTensorData* revTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:revBuf shape:@[@25] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[nationRevenue] = revTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: CPU post — sort and print via shared helper
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    postProcessQ5((float*)[revBuf contents], nation_names);
    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q5", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
