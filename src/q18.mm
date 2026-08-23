#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q18 — Large Volume Customer
//
// SELECT c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice, SUM(l_quantity)
// FROM customer, orders, lineitem
// WHERE o_orderkey IN (
//     SELECT l_orderkey FROM lineitem GROUP BY l_orderkey HAVING SUM(l_quantity) > 300)
//   AND c_custkey = o_custkey AND o_orderkey = l_orderkey
// GROUP BY c_name, c_custkey, o_orderkey, o_orderdate, o_totalprice
// ORDER BY o_totalprice DESC, o_orderdate LIMIT 100;
//
// Strategy:
//   MPSGraph:  one scatter-add pass over lineitem (the large table) computes
//              SUM(l_quantity) per orderkey directly — this *is* the HAVING subquery.
//   CPU post:  threshold at 300, join back to orders/customer (small tables), sort, print.

void runQ18(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q18: Large Volume Customer ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {0, ColType::INT  },   // l_orderkey
        {4, ColType::FLOAT},   // l_quantity
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_qty      = lCols.floats(4);
    size_t N = l_orderkey.size();

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT  },   // o_orderkey
        {1, ColType::INT  },   // o_custkey
        {3, ColType::FLOAT},   // o_totalprice
        {4, ColType::DATE },   // o_orderdate
    });
    auto& o_orderkey   = oCols.ints(0);
    auto& o_custkey    = oCols.ints(1);
    auto& o_totalprice = oCols.floats(3);
    auto& o_orderdate  = oCols.ints(4);

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {
        {0, ColType::INT        },
        {1, ColType::CHAR_FIXED, 25},   // c_name
    });
    auto& c_custkey = cCols.ints(0);
    auto& c_name    = cCols.chars(1);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Lineitem: %zu  Orders: %zu  Customer: %zu  (parse: %.1f ms)\n",
           N, o_orderkey.size(), c_custkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: Build MPSGraph — SUM(l_quantity) per orderkey
    // ----------------------------------------------------------------
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());

    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSGraphTensor* tOrderkey = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tQty      = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"l_qty"];

    MPSGraphTensor* zeros = [graph constantWithScalar:0.0 shape:@[@(max_orderkey + 1)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* sumQtyByOrder = [graph scatterWithDataTensor:zeros
                                                    updatesTensor:tQty
                                                    indicesTensor:tOrderkey
                                                             axis:0
                                                             mode:MPSGraphScatterModeAdd
                                                             name:@"sum_qty_by_order"];

    // ----------------------------------------------------------------
    // Step 3: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* okTD = columnTensor(device, (void*)l_orderkey.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* qtTD = columnTensor(device, (void*)l_qty.data(),      N, MPSDataTypeFloat32);
    NSDictionary* feeds = @{ tOrderkey: okTD, tQty: qtTD };

    id<MTLBuffer> sumBuf = allocSharedBuffer(device, (size_t)(max_orderkey + 1) * sizeof(float));
    MPSGraphTensorData* sumTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:sumBuf shape:@[@(max_orderkey + 1)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[sumQtyByOrder] = sumTD;

    // ----------------------------------------------------------------
    // Step 4: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 5: CPU post — threshold, join to orders/customer, sort, print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* sumQty = (float*)[sumBuf contents];

    int max_custkey = *std::max_element(c_custkey.begin(), c_custkey.end());
    std::vector<std::string> custName((size_t)(max_custkey + 1));
    for (size_t i = 0; i < c_custkey.size(); i++)
        custName[(size_t)c_custkey[i]] = trimFixed(c_name.data(), i, 25);

    struct ResultRow {
        std::string name;
        int custkey, orderkey, orderdate;
        float totalprice, qty;
    };
    std::vector<ResultRow> rows;
    for (size_t i = 0; i < o_orderkey.size(); i++) {
        int ok = o_orderkey[i];
        if (ok < 0 || ok > max_orderkey) continue;
        if (sumQty[ok] <= 300.0f) continue;
        int ck = o_custkey[i];
        std::string nm = (ck >= 0 && ck <= max_custkey) ? custName[(size_t)ck] : "";
        rows.push_back({nm, ck, ok, o_orderdate[i], o_totalprice[i], sumQty[ok]});
    }

    std::sort(rows.begin(), rows.end(), [](const ResultRow& a, const ResultRow& b) {
        if (a.totalprice != b.totalprice) return a.totalprice > b.totalprice;
        return a.orderdate < b.orderdate;
    });
    if (rows.size() > 100) rows.resize(100);

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q18 Results (first 15 of %zu, LIMIT 100):\n", rows.size());
    printf("+--------------------+----------+----------+------------+--------------+----------+\n");
    printf("| c_name             | custkey  | orderkey | orderdate  | totalprice   | sum_qty  |\n");
    printf("+--------------------+----------+----------+------------+--------------+----------+\n");
    size_t show = std::min(rows.size(), (size_t)15);
    for (size_t i = 0; i < show; i++)
        printf("| %-18s | %8d | %8d | %10d | %12.2f | %8.0f |\n",
               rows[i].name.c_str(), rows[i].custkey, rows[i].orderkey,
               rows[i].orderdate, rows[i].totalprice, rows[i].qty);
    printf("+--------------------+----------+----------+------------+--------------+----------+\n");

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q18", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
