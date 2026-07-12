#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q7 — Volume Shipping
//
// SELECT supp_nation, cust_nation, l_year, SUM(volume) AS revenue
// FROM supplier, lineitem, orders, customer, nation n1, nation n2
// WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey
//   AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey
//   AND ((n1.n_name='FRANCE' AND n2.n_name='GERMANY') OR (n1.n_name='GERMANY' AND n2.n_name='FRANCE'))
//   AND l_shipdate BETWEEN '1995-01-01' AND '1996-12-31'
// GROUP BY supp_nation, cust_nation, l_year
//
// Strategy (mirrors Q5's hybrid CPU-map + MPSGraph-gather approach):
//   CPU build: supp_nation_map[suppkey] = nationkey (FRANCE/GERMANY only, else -1);
//              orders_cust_nation_map[orderkey] = customer nationkey (FRANCE/GERMANY only,
//              else -1) — pre-joins orders+customer on CPU exactly as Q5 does.
//   MPSGraph:  gather both nations per lineitem row via l_suppkey/l_orderkey; mask on
//              validity + cross-nation pair + ship-date window; bucket into 4 fixed
//              (pair, year) groups via masked reductionSum (Q4-style).
//   CPU post:  print the 2x2 revenue table.

void runQ7(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q7: Volume Shipping ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto sup = loadSupplierBasic(g_dataset_path);
    auto& s_suppkey   = sup.suppkey;
    auto& s_nationkey = sup.nationkey;

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT}, {3, ColType::INT},   // c_custkey, c_nationkey
    });
    auto& c_custkey   = cCols.ints(0);
    auto& c_nationkey = cCols.ints(3);

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT}, {1, ColType::INT},   // o_orderkey, o_custkey
    });
    auto& o_orderkey = oCols.ints(0);
    auto& o_custkey  = oCols.ints(1);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {0, ColType::INT  }, {2, ColType::INT  },
        {5, ColType::FLOAT}, {6, ColType::FLOAT}, {10, ColType::DATE},
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_suppkey  = lCols.ints(2);
    auto& l_extprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto& l_shipdate = lCols.ints(10);
    size_t N = l_orderkey.size();

    auto nat = loadNation(g_dataset_path);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Supplier: %zu  Customer: %zu  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n",
           s_suppkey.size(), c_custkey.size(), o_orderkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — FRANCE/GERMANY nationkeys
    // ----------------------------------------------------------------
    int france_nk  = findNationKey(nat, "FRANCE");
    int germany_nk = findNationKey(nat, "GERMANY");
    if (france_nk == -1 || germany_nk == -1) {
        fprintf(stderr, "  FRANCE/GERMANY not found\n");
        return;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — build lookup maps
    // ----------------------------------------------------------------
    int max_suppkey  = *std::max_element(s_suppkey.begin(),  s_suppkey.end());
    int max_custkey  = *std::max_element(c_custkey.begin(),  c_custkey.end());
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());

    // supp_nation_map[suppkey] = nationkey if FRANCE/GERMANY, else -1
    std::vector<int32_t> supp_nation_map((size_t)(max_suppkey + 1), -1);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        int nk = s_nationkey[i];
        if (nk == france_nk || nk == germany_nk)
            supp_nation_map[(size_t)s_suppkey[i]] = nk;
    }

    // cust_nation_map[custkey] = nationkey if FRANCE/GERMANY, else -1
    std::vector<int32_t> cust_nation_map((size_t)(max_custkey + 1), -1);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        int nk = c_nationkey[i];
        if (nk == france_nk || nk == germany_nk)
            cust_nation_map[(size_t)c_custkey[i]] = nk;
    }

    // orders_cust_nation_map[orderkey] = customer nationkey (pre-joins orders+customer)
    std::vector<int32_t> orders_cust_nation_map((size_t)(max_orderkey + 1), -1);
    for (size_t i = 0; i < o_orderkey.size(); i++) {
        int ck = o_custkey[i];
        if (ck > max_custkey) continue;
        int nk = cust_nation_map[(size_t)ck];
        if (nk >= 0)
            orders_cust_nation_map[(size_t)o_orderkey[i]] = nk;
    }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tOrdCustNatMap = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                        dataType:MPSDataTypeInt32   name:@"ord_cust_nat_map"];
    MPSGraphTensor* tSuppNatMap    = [graph placeholderWithShape:@[@(max_suppkey + 1)]
                                                        dataType:MPSDataTypeInt32   name:@"supp_nat_map"];
    MPSGraphTensor* tLOrderkey     = [graph placeholderWithShape:@[@(N)]
                                                        dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tLSuppkey      = [graph placeholderWithShape:@[@(N)]
                                                        dataType:MPSDataTypeInt32   name:@"l_suppkey"];
    MPSGraphTensor* tShipdate      = [graph placeholderWithShape:@[@(N)]
                                                        dataType:MPSDataTypeInt32   name:@"l_shipdate"];
    MPSGraphTensor* tExtprice      = [graph placeholderWithShape:@[@(N)]
                                                        dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount      = [graph placeholderWithShape:@[@(N)]
                                                        dataType:MPSDataTypeFloat32 name:@"discount"];

    // Gather per-lineitem nationkeys via the orders/supplier maps
    MPSGraphTensor* custNation = [graph gatherWithUpdatesTensor:tOrdCustNatMap
                                                  indicesTensor:tLOrderkey
                                                           axis:0 batchDimensions:0 name:@"cust_nat"];
    MPSGraphTensor* suppNation = [graph gatherWithUpdatesTensor:tSuppNatMap
                                                  indicesTensor:tLSuppkey
                                                           axis:0 batchDimensions:0 name:@"supp_nat"];

    // Validity + cross-nation-pair condition (same-nation combos don't qualify)
    MPSGraphTensor* zeroI     = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* validCust = [graph greaterThanOrEqualToWithPrimaryTensor:custNation
                                                              secondaryTensor:zeroI name:nil];
    MPSGraphTensor* validSupp = [graph greaterThanOrEqualToWithPrimaryTensor:suppNation
                                                              secondaryTensor:zeroI name:nil];
    MPSGraphTensor* natDiffer = [graph notEqualWithPrimaryTensor:custNation
                                                  secondaryTensor:suppNation name:nil];

    // Ship-date window: [1995-01-01, 1996-12-31]
    MPSGraphTensor* dateStart = [graph constantWithScalar:19950101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateEnd   = [graph constantWithScalar:19961231 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateOk    = [graph logicalANDWithPrimaryTensor:
                                     [graph greaterThanOrEqualToWithPrimaryTensor:tShipdate
                                                                  secondaryTensor:dateStart name:nil]
                                 secondaryTensor:
                                     [graph lessThanOrEqualToWithPrimaryTensor:tShipdate
                                                              secondaryTensor:dateEnd name:nil]
                                 name:nil];

    MPSGraphTensor* mask = [graph logicalANDWithPrimaryTensor:
                                [graph logicalANDWithPrimaryTensor:validCust secondaryTensor:validSupp name:nil]
                            secondaryTensor:
                                [graph logicalANDWithPrimaryTensor:natDiffer secondaryTensor:dateOk name:nil]
                            name:@"mask"];
    MPSGraphTensor* maskF = [graph castTensor:mask toType:MPSDataTypeFloat32 name:nil];

    // volume = extprice * (1 - discount) * mask
    MPSGraphTensor* oneF   = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* volume = [graph multiplicationWithPrimaryTensor:
                                  [graph multiplicationWithPrimaryTensor:tExtprice
                                      secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                          secondaryTensor:tDiscount name:nil]
                                      name:nil]
                              secondaryTensor:maskF name:@"volume"];

    // pair p: 0 => (supp=FRANCE, cust=GERMANY), 1 => (supp=GERMANY, cust=FRANCE)
    // Always 0/1 regardless of mask (comparison-derived), so it's a safe scatter index.
    MPSGraphTensor* germanyC = [graph constantWithScalar:germany_nk dataType:MPSDataTypeInt32];
    MPSGraphTensor* pairIdx  = [graph castTensor:
                                    [graph equalWithPrimaryTensor:suppNation secondaryTensor:germanyC name:nil]
                                        toType:MPSDataTypeInt32 name:nil];

    // year flag y: 0 => 1995, 1 => 1996
    MPSGraphTensor* year1996Start = [graph constantWithScalar:19960101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* yearIdx = [graph castTensor:
                                   [graph greaterThanOrEqualToWithPrimaryTensor:tShipdate
                                                                secondaryTensor:year1996Start name:nil]
                                       toType:MPSDataTypeInt32 name:nil];

    MPSGraphTensor* two      = [graph constantWithScalar:2 dataType:MPSDataTypeInt32];
    MPSGraphTensor* groupIdx = [graph additionWithPrimaryTensor:
                                    [graph multiplicationWithPrimaryTensor:pairIdx secondaryTensor:two name:nil]
                                secondaryTensor:yearIdx name:@"group_idx"];

    // Scatter volume into 4-element (pair, year) revenue array
    MPSGraphTensor* zeros4  = [graph constantWithScalar:0.0 shape:@[@4] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue = [graph scatterWithDataTensor:zeros4
                                             updatesTensor:volume
                                             indicesTensor:groupIdx
                                                      axis:0
                                                      mode:MPSGraphScatterModeAdd
                                                      name:@"revenue"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* ocnTD = columnTensor(device, orders_cust_nation_map.data(),
                                             (size_t)(max_orderkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* snTD  = columnTensor(device, supp_nation_map.data(),
                                             (size_t)(max_suppkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* okTD  = columnTensor(device, (void*)l_orderkey.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* skTD  = columnTensor(device, (void*)l_suppkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* sdTD  = columnTensor(device, (void*)l_shipdate.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* epTD  = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD  = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tOrdCustNatMap: ocnTD, tSuppNatMap: snTD,
        tLOrderkey: okTD,      tLSuppkey:   skTD, tShipdate: sdTD,
        tExtprice:  epTD,      tDiscount:   diTD,
    };

    id<MTLBuffer> revBuf = allocSharedBuffer(device, 4 * sizeof(float));
    MPSGraphTensorData* revTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:revBuf shape:@[@4] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[revenue] = revTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: CPU post — print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* bins = (float*)[revBuf contents];

    printf("\nTPC-H Q7 Results:\n");
    printf("+----------+----------+--------+-----------------+\n");
    printf("| supp_nat | cust_nat | l_year |         revenue |\n");
    printf("+----------+----------+--------+-----------------+\n");
    static const char* pair_supp[] = {"FRANCE", "GERMANY"};
    static const char* pair_cust[] = {"GERMANY", "FRANCE"};
    for (int p = 0; p < 2; p++) {
        for (int y = 0; y < 2; y++) {
            printf("| %-8s | %-8s | %6d | $%14.2f |\n",
                   pair_supp[p], pair_cust[p], 1995 + y, bins[p * 2 + y]);
        }
    }
    printf("+----------+----------+--------+-----------------+\n");

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q7", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
