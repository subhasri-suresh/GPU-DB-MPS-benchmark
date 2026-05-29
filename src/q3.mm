#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q3 — Shipping Priority
//
// SELECT l_orderkey, SUM(l_extendedprice*(1-l_discount)) AS revenue,
//        o_orderdate, o_shippriority
// FROM customer, orders, lineitem
// WHERE c_mktsegment = 'BUILDING'
//   AND c_custkey    = o_custkey
//   AND l_orderkey   = o_orderkey
//   AND o_orderdate  < '1995-03-15'
//   AND l_shipdate   > '1995-03-15'
// GROUP BY l_orderkey, o_orderdate, o_shippriority
// ORDER BY revenue DESC, o_orderdate
// LIMIT 10
//
// Hybrid strategy:
//   CPU build: customer bitmap (BUILDING), orders direct-map (orderkey → dense idx).
//   MPSGraph:  gather order-idx per lineitem, filter shipdate, scatter revenue into
//              dense per-qualifying-order array.
//   CPU post:  sort top-10 by revenue DESC / orderdate ASC.

void runQ3(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q3: Shipping Priority ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT  },   // c_custkey
        {6, ColType::CHAR1},   // c_mktsegment (first char; 'B' = BUILDING)
    });
    auto& c_custkey    = cCols.ints(0);
    auto& c_mktsegment = cCols.chars(6);

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT },    // o_orderkey
        {1, ColType::INT },    // o_custkey
        {4, ColType::DATE},    // o_orderdate
        {7, ColType::INT },    // o_shippriority
    });
    auto& o_orderkey     = oCols.ints(0);
    auto& o_custkey      = oCols.ints(1);
    auto& o_orderdate    = oCols.ints(4);
    auto& o_shippriority = oCols.ints(7);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 0, ColType::INT  },   // l_orderkey
        { 5, ColType::FLOAT},   // l_extendedprice
        { 6, ColType::FLOAT},   // l_discount
        {10, ColType::DATE },   // l_shipdate
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_extprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    auto& l_shipdate = lCols.ints(10);
    size_t N = l_orderkey.size();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Customer: %zu  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n",
           c_custkey.size(), o_orderkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — customer bitmap for 'BUILDING' (first char 'B')
    // ----------------------------------------------------------------
    int max_custkey = *std::max_element(c_custkey.begin(), c_custkey.end());
    std::vector<uint8_t> cust_bitmap((size_t)(max_custkey + 1), 0);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        if (c_mktsegment[i] == 'B')
            cust_bitmap[(size_t)c_custkey[i]] = 1;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — orders direct-map: orderkey → dense qualifying index
    //   Qualifying: o_orderdate < 19950315 AND customer is BUILDING
    // ----------------------------------------------------------------
    const int CUTOFF = 19950315;
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());
    std::vector<int32_t> orders_map((size_t)(max_orderkey + 1), -1);

    std::vector<int32_t> q_orderkey_list, q_orderdate_list, q_shippriority_list;
    q_orderkey_list.reserve(o_orderkey.size() / 4);
    q_orderdate_list.reserve(o_orderkey.size() / 4);
    q_shippriority_list.reserve(o_orderkey.size() / 4);

    for (size_t i = 0; i < o_orderkey.size(); i++) {
        int ck = o_custkey[i];
        if (ck > max_custkey || !cust_bitmap[(size_t)ck]) continue;
        if (o_orderdate[i] >= CUTOFF) continue;
        int32_t idx = (int32_t)q_orderkey_list.size();
        orders_map[(size_t)o_orderkey[i]] = idx;
        q_orderkey_list.push_back(o_orderkey[i]);
        q_orderdate_list.push_back(o_orderdate[i]);
        q_shippriority_list.push_back(o_shippriority[i]);
    }
    size_t num_q = q_orderkey_list.size();
    printf("  Qualifying orders: %zu\n", num_q);
    if (num_q == 0) { printf("  No qualifying orders.\n"); return; }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tMap      = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                   dataType:MPSDataTypeInt32   name:@"orders_map"];
    MPSGraphTensor* tOrdKey   = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tShipdate = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeInt32   name:@"shipdate"];
    MPSGraphTensor* tExtprice = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount = [graph placeholderWithShape:@[@(N)]
                                                   dataType:MPSDataTypeFloat32 name:@"discount"];

    // Gather qualifying-order dense index for each lineitem row
    MPSGraphTensor* gatheredIdx = [graph gatherWithUpdatesTensor:tMap
                                                  indicesTensor:tOrdKey
                                                           axis:0
                                                batchDimensions:0
                                                           name:@"gathered_idx"];

    // valid_order: gathered index >= 0
    MPSGraphTensor* zeroI      = [graph constantWithScalar:0   dataType:MPSDataTypeInt32];
    MPSGraphTensor* cutoffI    = [graph constantWithScalar:CUTOFF dataType:MPSDataTypeInt32];
    MPSGraphTensor* validOrder = [graph greaterThanOrEqualToWithPrimaryTensor:gatheredIdx
                                                              secondaryTensor:zeroI name:nil];
    // valid_shipdate: l_shipdate > CUTOFF
    MPSGraphTensor* validShip  = [graph greaterThanWithPrimaryTensor:tShipdate
                                                     secondaryTensor:cutoffI name:nil];
    // Combined mask (bool → float)
    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:validOrder
                                                    secondaryTensor:validShip name:nil]
                             toType:MPSDataTypeFloat32 name:@"mask"];

    // Revenue = extprice * (1 - discount) * mask
    MPSGraphTensor* oneF    = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue = [graph multiplicationWithPrimaryTensor:
                                   [graph multiplicationWithPrimaryTensor:tExtprice
                                       secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                           secondaryTensor:tDiscount name:nil]
                                       name:nil]
                               secondaryTensor:maskF name:@"revenue"];

    // Clamp index to ≥ 0 so invalid rows scatter harmlessly into bucket 0 (value = 0)
    MPSGraphTensor* clampedIdx = [graph maximumWithPrimaryTensor:gatheredIdx
                                                secondaryTensor:zeroI name:@"clamped_idx"];

    // Scatter-add revenue per qualifying order
    MPSGraphTensor* zeros       = [graph constantWithScalar:0.0 shape:@[@(num_q)]
                                                   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* orderRevenue = [graph scatterWithDataTensor:zeros
                                                 updatesTensor:revenue
                                                 indicesTensor:clampedIdx
                                                          axis:0
                                                          mode:MPSGraphScatterModeAdd
                                                          name:@"order_revenue"];

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* mapTD = columnTensor(device, orders_map.data(),
                                             (size_t)(max_orderkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* okTD  = columnTensor(device, (void*)l_orderkey.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* sdTD  = columnTensor(device, (void*)l_shipdate.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* epTD  = columnTensor(device, (void*)l_extprice.data(),  N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD  = columnTensor(device, (void*)l_discount.data(),  N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tMap: mapTD, tOrdKey: okTD, tShipdate: sdTD, tExtprice: epTD, tDiscount: diTD
    };

    id<MTLBuffer> revBuf = allocSharedBuffer(device, num_q * sizeof(float));
    MPSGraphTensorData* revTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:revBuf shape:@[@(num_q)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[orderRevenue] = revTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: CPU post — collect non-zero entries, sort, top-10
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* rev = (float*)[revBuf contents];

    struct Q3Row { int32_t orderkey; float revenue; int32_t orderdate; int32_t shippriority; };
    std::vector<Q3Row> rows;
    rows.reserve(num_q / 2);
    for (size_t i = 0; i < num_q; i++) {
        if (rev[i] > 0.0f)
            rows.push_back({q_orderkey_list[i], rev[i], q_orderdate_list[i], q_shippriority_list[i]});
    }
    size_t topK = std::min(rows.size(), (size_t)10);
    std::partial_sort(rows.begin(), rows.begin() + (ptrdiff_t)topK, rows.end(),
        [](const Q3Row& a, const Q3Row& b) {
            if (a.revenue != b.revenue) return a.revenue > b.revenue;
            return a.orderdate < b.orderdate;
        });

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q3 Results (Top 10):\n");
    printf("+----------+----------------+------------+--------------+\n");
    printf("| orderkey |        revenue | orderdate  | shippriority |\n");
    printf("+----------+----------------+------------+--------------+\n");
    for (size_t i = 0; i < topK; i++) {
        printf("| %8d | %14.2f | %10d | %12d |\n",
               rows[i].orderkey, rows[i].revenue,
               rows[i].orderdate, rows[i].shippriority);
    }
    printf("+----------+----------------+------------+--------------+\n");

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q3", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
