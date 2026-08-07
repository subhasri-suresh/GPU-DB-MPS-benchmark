#import "mps_infra.h"
#import <Foundation/Foundation.h>
#import <algorithm>

// TPC-H Q10 — Returned Item Reporting
//
// SELECT c_custkey, c_name, SUM(l_extendedprice*(1-l_discount)) AS revenue,
//        c_acctbal, n_name, c_address, c_phone, c_comment
// FROM customer, orders, lineitem, nation
// WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey
//   AND o_orderdate >= '1993-10-01' AND o_orderdate < '1994-01-01'
//   AND l_returnflag = 'R' AND c_nationkey = n_nationkey
// GROUP BY c_custkey, ... ORDER BY revenue DESC LIMIT 20
//
// Strategy (mirrors the raw-Metal baseline's build-orders-map + probe approach):
//   CPU build: orders_map[orderkey] = custkey if order date in the 3-month window,
//              else -1 (pre-joins orders on date; customer join happens implicitly
//              since custkey itself is the group key).
//   CPU:       returnFlagMask[i] = 1.0 if l_returnflag[i] == 'R' (char columns aren't
//              fed to MPSGraph, so the comparison is precomputed like Q12's shipmodeMask).
//   MPSGraph:  gather custkey per lineitem row; mask on valid order + return flag;
//              scatter revenue into a dense (max_custkey+1)-sized per-customer array.
//   CPU post:  bounded 20-element min-heap over the dense array, then print.

void runQ10(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q10: Returned Item Reporting ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT}, {1, ColType::CHAR_FIXED, 18}, {2, ColType::CHAR_FIXED, 40},
        {3, ColType::INT}, {4, ColType::CHAR_FIXED, 15}, {5, ColType::FLOAT},
        {7, ColType::CHAR_FIXED, 117},
    });
    auto& c_custkey   = cCols.ints(0);
    auto& c_name      = cCols.chars(1);
    auto& c_address   = cCols.chars(2);
    auto& c_nationkey = cCols.ints(3);
    auto& c_phone     = cCols.chars(4);
    auto& c_acctbal   = cCols.floats(5);
    auto& c_comment   = cCols.chars(7);
    (void)c_address; (void)c_phone; (void)c_comment;   // loaded for parity, unused in printed table

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT}, {1, ColType::INT}, {4, ColType::DATE},
    });
    auto& o_orderkey  = oCols.ints(0);
    auto& o_custkey   = oCols.ints(1);
    auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {0, ColType::INT}, {5, ColType::FLOAT}, {6, ColType::FLOAT}, {8, ColType::CHAR1},
    });
    auto& l_orderkey    = lCols.ints(0);
    auto& l_extprice    = lCols.floats(5);
    auto& l_discount    = lCols.floats(6);
    auto& l_returnflag  = lCols.chars(8);
    size_t N = l_orderkey.size();

    auto nat = loadNation(g_dataset_path);
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Customer: %zu  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n",
           c_custkey.size(), o_orderkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — custkey → row index (for post-process lookups)
    // ----------------------------------------------------------------
    int max_custkey = *std::max_element(c_custkey.begin(), c_custkey.end());
    std::vector<size_t> cust_index((size_t)(max_custkey + 1), SIZE_MAX);
    for (size_t i = 0; i < c_custkey.size(); i++)
        cust_index[(size_t)c_custkey[i]] = i;

    // ----------------------------------------------------------------
    // Step 3: CPU — orders_map[orderkey] = custkey if date in [1993-10-01, 1994-01-01)
    // ----------------------------------------------------------------
    const int DATE_START = 19931001, DATE_END = 19940101;
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());
    std::vector<int32_t> orders_map((size_t)(max_orderkey + 1), -1);
    for (size_t i = 0; i < o_orderkey.size(); i++) {
        if (o_orderdate[i] < DATE_START || o_orderdate[i] >= DATE_END) continue;
        orders_map[(size_t)o_orderkey[i]] = o_custkey[i];
    }

    // ----------------------------------------------------------------
    // Step 4: CPU — return-flag mask per lineitem row (char comparison precomputed)
    // ----------------------------------------------------------------
    std::vector<float> return_mask(N);
    for (size_t i = 0; i < N; i++)
        return_mask[i] = (l_returnflag[i] == 'R') ? 1.0f : 0.0f;

    // ----------------------------------------------------------------
    // Step 5: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tOrdersMap = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                    dataType:MPSDataTypeInt32   name:@"orders_map"];
    MPSGraphTensor* tLOrderkey = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tRetMask   = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeFloat32 name:@"return_mask"];
    MPSGraphTensor* tExtprice  = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount  = [graph placeholderWithShape:@[@(N)]
                                                    dataType:MPSDataTypeFloat32 name:@"discount"];

    // Gather per-lineitem custkey via the date-filtered orders map
    MPSGraphTensor* custkey = [graph gatherWithUpdatesTensor:tOrdersMap
                                               indicesTensor:tLOrderkey
                                                        axis:0 batchDimensions:0 name:@"custkey"];

    MPSGraphTensor* zeroI      = [graph constantWithScalar:0 dataType:MPSDataTypeInt32];
    MPSGraphTensor* validOrder = [graph greaterThanOrEqualToWithPrimaryTensor:custkey
                                                              secondaryTensor:zeroI name:nil];
    MPSGraphTensor* maskF = [graph multiplicationWithPrimaryTensor:
                                 [graph castTensor:validOrder toType:MPSDataTypeFloat32 name:nil]
                             secondaryTensor:tRetMask name:@"mask"];

    // revenue = extprice * (1 - discount) * mask
    MPSGraphTensor* oneF    = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue = [graph multiplicationWithPrimaryTensor:
                                   [graph multiplicationWithPrimaryTensor:tExtprice
                                       secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                           secondaryTensor:tDiscount name:nil]
                                       name:nil]
                               secondaryTensor:maskF name:@"revenue"];

    // Clamp to a safe scatter index; invalid rows carry 0 revenue so bucket 0 is unaffected
    MPSGraphTensor* clampedCust = [graph maximumWithPrimaryTensor:custkey
                                                  secondaryTensor:zeroI name:@"clamped_cust"];

    MPSGraphTensor* zerosCust = [graph constantWithScalar:0.0 shape:@[@(max_custkey + 1)]
                                                 dataType:MPSDataTypeFloat32];
    MPSGraphTensor* custRevenue = [graph scatterWithDataTensor:zerosCust
                                                updatesTensor:revenue
                                                indicesTensor:clampedCust
                                                         axis:0
                                                         mode:MPSGraphScatterModeAdd
                                                         name:@"cust_revenue"];

    // ----------------------------------------------------------------
    // Step 6: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* omTD = columnTensor(device, orders_map.data(),
                                            (size_t)(max_orderkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* okTD = columnTensor(device, (void*)l_orderkey.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* rmTD = columnTensor(device, return_mask.data(),       N, MPSDataTypeFloat32);
    MPSGraphTensorData* epTD = columnTensor(device, (void*)l_extprice.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tOrdersMap: omTD, tLOrderkey: okTD, tRetMask: rmTD,
        tExtprice:  epTD, tDiscount:  diTD,
    };

    id<MTLBuffer> revBuf = allocSharedBuffer(device, (size_t)(max_custkey + 1) * sizeof(float));
    MPSGraphTensorData* revTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:revBuf shape:@[@(max_custkey + 1)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[custRevenue] = revTD;

    // ----------------------------------------------------------------
    // Step 7: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 8: CPU post — top 20 via bounded min-heap
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* cust_revenue = (float*)[revBuf contents];

    struct Q10Result { int custkey; float revenue; };
    auto cmp = [](const Q10Result& a, const Q10Result& b) { return a.revenue > b.revenue; };
    std::vector<Q10Result> topHeap;
    topHeap.reserve(21);
    for (int i = 0; i <= max_custkey; i++) {
        if (cust_revenue[i] > 0.0f) {
            if (topHeap.size() < 20) {
                topHeap.push_back({i, cust_revenue[i]});
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            } else if (cust_revenue[i] > topHeap.front().revenue) {
                std::pop_heap(topHeap.begin(), topHeap.end(), cmp);
                topHeap.back() = {i, cust_revenue[i]};
                std::push_heap(topHeap.begin(), topHeap.end(), cmp);
            }
        }
    }
    std::sort_heap(topHeap.begin(), topHeap.end(), cmp);

    printf("\nTPC-H Q10 Results (Top 20):\n");
    printf("+---------+------------------+------------+----------+------------------+\n");
    printf("| custkey |           c_name |    revenue | c_acctbal|           n_name |\n");
    printf("+---------+------------------+------------+----------+------------------+\n");
    for (size_t i = 0; i < topHeap.size(); i++) {
        int ck = topHeap[i].custkey;
        size_t ci = cust_index[(size_t)ck];
        if (ci == SIZE_MAX) continue;
        printf("| %7d | %-16s | $%10.2f| %8.2f | %-16s |\n",
               ck, trimFixed(c_name.data(), ci, 18).c_str(),
               topHeap[i].revenue, c_acctbal[ci],
               nation_names[c_nationkey[ci]].c_str());
    }
    printf("+---------+------------------+------------+----------+------------------+\n");

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q10", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
