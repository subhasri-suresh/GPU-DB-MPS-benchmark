#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q8 — National Market Share
//
// SELECT o_year, SUM(CASE WHEN nation='BRAZIL' THEN volume ELSE 0 END) / SUM(volume) AS mkt_share
// FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region
// WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey
//   AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey AND n1.n_regionkey = r_regionkey
//   AND r_name = 'AMERICA' AND s_nationkey = n2.n_nationkey
//   AND o_orderdate BETWEEN '1995-01-01' AND '1996-12-31'
//   AND p_type = 'ECONOMY ANODIZED STEEL'
// GROUP BY o_year
//
// Strategy (mirrors the raw-Metal baseline's hybrid CPU-map + probe approach):
//   CPU build: part_match[partkey] = 1.0 if p_type matches; cust_nation_map[custkey] =
//              nationkey if customer is in AMERICA, else -1; orders_year_map[orderkey] =
//              year (1995/1996) if order date in range AND customer is AMERICA, else -1
//              (pre-joins orders+customer, same as the raw-Metal build-orders-map kernel);
//              supp_nation_map[suppkey] = nationkey (all suppliers).
//   MPSGraph:  gather part/order/supplier lookups per lineitem row; mask on part match +
//              valid year; scatter revenue into 2 (year) buckets for the grand total, and
//              a second masked scatter (supplier nation == BRAZIL) for the Brazil-only total.
//   CPU post:  mkt_share[y] = brazil[y] / total[y].

void runQ8(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q8: National Market Share ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT}, {4, ColType::CHAR_FIXED, 25},   // p_partkey, p_type
    });
    auto& p_partkey = pCols.ints(0);
    auto& p_type    = pCols.chars(4);

    auto sup = loadSupplierBasic(g_dataset_path);
    auto& s_suppkey   = sup.suppkey;
    auto& s_nationkey = sup.nationkey;

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT}, {3, ColType::INT},   // c_custkey, c_nationkey
    });
    auto& c_custkey   = cCols.ints(0);
    auto& c_nationkey = cCols.ints(3);

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT }, {1, ColType::INT }, {4, ColType::DATE},
    });
    auto& o_orderkey  = oCols.ints(0);
    auto& o_custkey   = oCols.ints(1);
    auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {0, ColType::INT  }, {1, ColType::INT  }, {2, ColType::INT  },
        {5, ColType::FLOAT}, {6, ColType::FLOAT},
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_partkey  = lCols.ints(1);
    auto& l_suppkey  = lCols.ints(2);
    auto& l_extprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    size_t N = l_orderkey.size();

    auto nat = loadNation(g_dataset_path, /*with_regionkey=*/true);
    auto reg = loadRegion(g_dataset_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  Supplier: %zu  Customer: %zu  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n",
           p_partkey.size(), s_suppkey.size(), c_custkey.size(), o_orderkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — AMERICA region, BRAZIL nation
    // ----------------------------------------------------------------
    int america_rk = findRegionKey(reg.regionkey, reg.name.data(), RegionData::NAME_WIDTH, "AMERICA");
    int brazil_nk  = findNationKey(nat, "BRAZIL");
    if (america_rk == -1 || brazil_nk == -1) {
        fprintf(stderr, "  AMERICA/BRAZIL not found\n");
        return;
    }
    uint america_bitmap = buildNationBitmap(nat.nationkey, nat.regionkey, america_rk);

    // ----------------------------------------------------------------
    // Step 3: CPU — build lookup maps
    // ----------------------------------------------------------------
    int max_partkey  = *std::max_element(p_partkey.begin(), p_partkey.end());
    int max_suppkey  = *std::max_element(s_suppkey.begin(), s_suppkey.end());
    int max_custkey  = *std::max_element(c_custkey.begin(), c_custkey.end());
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());

    // part_match[partkey] = 1.0 if p_type == 'ECONOMY ANODIZED STEEL'
    std::vector<float> part_match((size_t)(max_partkey + 1), 0.0f);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        if (trimFixed(p_type.data(), i, 25) == "ECONOMY ANODIZED STEEL")
            part_match[(size_t)p_partkey[i]] = 1.0f;
    }

    // supp_nation_map[suppkey] = nationkey (all suppliers)
    std::vector<int32_t> supp_nation_map((size_t)(max_suppkey + 1), -1);
    for (size_t i = 0; i < s_suppkey.size(); i++)
        supp_nation_map[(size_t)s_suppkey[i]] = s_nationkey[i];

    // cust_nation_map[custkey] = nationkey if AMERICA, else -1
    std::vector<int32_t> cust_nation_map((size_t)(max_custkey + 1), -1);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        int nk = c_nationkey[i];
        if (nk >= 0 && nk < 25 && ((america_bitmap >> nk) & 1))
            cust_nation_map[(size_t)c_custkey[i]] = nk;
    }

    // orders_year_map[orderkey] = year (1995/1996) if date in range AND AMERICA customer, else -1
    const int DATE_START = 19950101, DATE_END = 19961231;
    std::vector<int32_t> orders_year_map((size_t)(max_orderkey + 1), -1);
    for (size_t i = 0; i < o_orderkey.size(); i++) {
        if (o_orderdate[i] < DATE_START || o_orderdate[i] > DATE_END) continue;
        int ck = o_custkey[i];
        if (ck > max_custkey || cust_nation_map[(size_t)ck] < 0) continue;
        orders_year_map[(size_t)o_orderkey[i]] = o_orderdate[i] / 10000;
    }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tPartMatch  = [graph placeholderWithShape:@[@(max_partkey + 1)]
                                                     dataType:MPSDataTypeFloat32 name:@"part_match"];
    MPSGraphTensor* tOrdYearMap = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                     dataType:MPSDataTypeInt32   name:@"ord_year_map"];
    MPSGraphTensor* tSuppNatMap = [graph placeholderWithShape:@[@(max_suppkey + 1)]
                                                     dataType:MPSDataTypeInt32   name:@"supp_nat_map"];
    MPSGraphTensor* tLPartkey   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeInt32   name:@"l_partkey"];
    MPSGraphTensor* tLOrderkey  = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tLSuppkey   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeInt32   name:@"l_suppkey"];
    MPSGraphTensor* tExtprice   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount   = [graph placeholderWithShape:@[@(N)]
                                                     dataType:MPSDataTypeFloat32 name:@"discount"];

    // Gather per-lineitem lookups
    MPSGraphTensor* partMatchF = [graph gatherWithUpdatesTensor:tPartMatch
                                                  indicesTensor:tLPartkey
                                                           axis:0 batchDimensions:0 name:@"part_match_f"];
    MPSGraphTensor* orderYear  = [graph gatherWithUpdatesTensor:tOrdYearMap
                                                  indicesTensor:tLOrderkey
                                                           axis:0 batchDimensions:0 name:@"order_year"];
    MPSGraphTensor* suppNation = [graph gatherWithUpdatesTensor:tSuppNatMap
                                                  indicesTensor:tLSuppkey
                                                           axis:0 batchDimensions:0 name:@"supp_nat"];

    // Validity: part type matches AND order year is valid (>= 1995 sentinel check)
    MPSGraphTensor* halfF     = [graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* partBool  = [graph greaterThanWithPrimaryTensor:partMatchF secondaryTensor:halfF name:nil];
    MPSGraphTensor* year1995  = [graph constantWithScalar:1995 dataType:MPSDataTypeInt32];
    MPSGraphTensor* validYear = [graph greaterThanOrEqualToWithPrimaryTensor:orderYear
                                                              secondaryTensor:year1995 name:nil];
    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:partBool secondaryTensor:validYear name:nil]
                             toType:MPSDataTypeFloat32 name:@"mask"];

    // revenue = extprice * (1 - discount) * mask
    MPSGraphTensor* oneF    = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue = [graph multiplicationWithPrimaryTensor:
                                   [graph multiplicationWithPrimaryTensor:tExtprice
                                       secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                           secondaryTensor:tDiscount name:nil]
                                       name:nil]
                               secondaryTensor:maskF name:@"revenue"];

    // Brazil-only revenue = revenue * (suppNation == BRAZIL)
    MPSGraphTensor* brazilC     = [graph constantWithScalar:brazil_nk dataType:MPSDataTypeInt32];
    MPSGraphTensor* brazilMaskF = [graph castTensor:
                                       [graph equalWithPrimaryTensor:suppNation secondaryTensor:brazilC name:nil]
                                       toType:MPSDataTypeFloat32 name:nil];
    MPSGraphTensor* brazilRevenue = [graph multiplicationWithPrimaryTensor:revenue
                                                          secondaryTensor:brazilMaskF name:@"brazil_revenue"];

    // year_idx = clamp(order_year - 1995, 0, 1) — safe scatter index; masked rows carry 0 revenue
    MPSGraphTensor* zeroI    = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* oneI     = [graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* yearIdx  = [graph subtractionWithPrimaryTensor:orderYear secondaryTensor:year1995 name:nil];
    MPSGraphTensor* clampedYearIdx = [graph maximumWithPrimaryTensor:
                                          [graph minimumWithPrimaryTensor:yearIdx secondaryTensor:oneI name:nil]
                                      secondaryTensor:zeroI name:@"clamped_year_idx"];

    // Scatter into 2-element (year) buckets: one for grand total, one for Brazil-only
    MPSGraphTensor* zeros2 = [graph constantWithScalar:0.0 shape:@[@2] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* totalByYear  = [graph scatterWithDataTensor:zeros2 updatesTensor:revenue
                                                  indicesTensor:clampedYearIdx axis:0
                                                           mode:MPSGraphScatterModeAdd name:@"total_by_year"];
    MPSGraphTensor* brazilByYear = [graph scatterWithDataTensor:zeros2 updatesTensor:brazilRevenue
                                                  indicesTensor:clampedYearIdx axis:0
                                                           mode:MPSGraphScatterModeAdd name:@"brazil_by_year"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffers
    // ----------------------------------------------------------------
    MPSGraphTensorData* pmTD = columnTensor(device, part_match.data(),
                                            (size_t)(max_partkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* oyTD = columnTensor(device, orders_year_map.data(),
                                            (size_t)(max_orderkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* snTD = columnTensor(device, supp_nation_map.data(),
                                            (size_t)(max_suppkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* pkTD = columnTensor(device, (void*)l_partkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* okTD = columnTensor(device, (void*)l_orderkey.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* skTD = columnTensor(device, (void*)l_suppkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* epTD = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tPartMatch: pmTD, tOrdYearMap: oyTD, tSuppNatMap: snTD,
        tLPartkey:  pkTD, tLOrderkey:  okTD, tLSuppkey:   skTD,
        tExtprice:  epTD, tDiscount:   diTD,
    };

    id<MTLBuffer> totalBuf  = allocSharedBuffer(device, 2 * sizeof(float));
    id<MTLBuffer> brazilBuf = allocSharedBuffer(device, 2 * sizeof(float));
    MPSGraphTensorData* totalTD  = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:totalBuf  shape:@[@2] dataType:MPSDataTypeFloat32];
    MPSGraphTensorData* brazilTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:brazilBuf shape:@[@2] dataType:MPSDataTypeFloat32];

    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[totalByYear]  = totalTD;
    results[brazilByYear] = brazilTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: CPU post — mkt_share = brazil / total
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* total  = (float*)[totalBuf  contents];
    float* brazil = (float*)[brazilBuf contents];

    printf("\nTPC-H Q8 Results:\n");
    printf("+--------+------------+\n");
    printf("| o_year |  mkt_share |\n");
    printf("+--------+------------+\n");
    for (int y = 0; y < 2; y++) {
        float mkt_share = (total[y] > 0.0f) ? brazil[y] / total[y] : 0.0f;
        printf("| %6d | %10.6f |\n", 1995 + y, mkt_share);
    }
    printf("+--------+------------+\n");

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q8", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
