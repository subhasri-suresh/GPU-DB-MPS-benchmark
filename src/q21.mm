#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q21 — Suppliers Who Kept Orders Waiting
//
// SELECT s_name, COUNT(*) AS numwait
// FROM supplier, lineitem l1, orders, nation
// WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey AND o_orderstatus = 'F'
//   AND l1.l_receiptdate > l1.l_commitdate
//   AND EXISTS (SELECT * FROM lineitem l2
//               WHERE l2.l_orderkey = l1.l_orderkey AND l2.l_suppkey <> l1.l_suppkey)
//   AND NOT EXISTS (SELECT * FROM lineitem l3
//                   WHERE l3.l_orderkey = l1.l_orderkey AND l3.l_suppkey <> l1.l_suppkey
//                     AND l3.l_receiptdate > l3.l_commitdate)
//   AND s_nationkey = n_nationkey AND n_name = 'SAUDI ARABIA'
// GROUP BY s_name ORDER BY numwait DESC, s_name LIMIT 100;
//
// Strategy: this is the classic hardest TPC-H query — two self-joins on lineitem keyed by
// (orderkey, supplier) composites that have no fixed cardinality like partsupp's 4-per-part,
// so there's no dense-array trick available (this is exactly the scenario the project plan
// flags as needing a "hybrid Metal+MPS" approach; MPSGraph alone has no hash-join primitive).
//   MPSGraph:  the one large-N elementwise pass — receiptdate > commitdate — over all of
//              lineitem, which is legitimately GPU-shaped work.
//   CPU:       TPC-H's dbgen guarantees lineitem rows for the same order are contiguous
//              (orders are generated, and their lineitems emitted, in orderkey order), and
//              an order has at most 7 lineitems — so a single linear scan groups each
//              order's rows into a tiny local buffer with no hash table. Per group: EXISTS
//              l2 reduces to "more than one distinct supplier in the order"; NOT EXISTS l3
//              reduces to "every late lineitem in the order shares l1's supplier" — both
//              O(group size) checks, no join structure needed.

void runQ21(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q21: Suppliers Who Kept Orders Waiting ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 0, ColType::INT },   // l_orderkey
        { 2, ColType::INT },   // l_suppkey
        {11, ColType::DATE},   // l_commitdate
        {12, ColType::DATE},   // l_receiptdate
    });
    auto& l_orderkey    = lCols.ints(0);
    auto& l_suppkey     = lCols.ints(2);
    auto& l_commitdate  = lCols.ints(11);
    auto& l_receiptdate = lCols.ints(12);
    size_t N = l_orderkey.size();

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT  },   // o_orderkey
        {2, ColType::CHAR1},   // o_orderstatus
    });
    auto& o_orderkey    = oCols.ints(0);
    auto& o_orderstatus = oCols.chars(2);

    auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl", {
        {0, ColType::INT        },
        {1, ColType::CHAR_FIXED, 25},   // s_name
        {3, ColType::INT        },      // s_nationkey
    });
    auto& s_suppkey   = sCols.ints(0);
    auto& s_name      = sCols.chars(1);
    auto& s_nationkey = sCols.ints(3);

    auto nat = loadNation(g_dataset_path);
    int target_nation = findNationKey(nat, "SAUDI ARABIA");

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Lineitem: %zu  Orders: %zu  Supplier: %zu  (parse: %.1f ms)\n",
           N, o_orderkey.size(), s_suppkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: Build MPSGraph — the one large-N pass: late = receiptdate > commitdate
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];
    MPSGraphTensor* tCommit  = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32 name:@"commitdate"];
    MPSGraphTensor* tReceipt = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32 name:@"receiptdate"];
    MPSGraphTensor* lateF = [graph castTensor:
                                 [graph greaterThanWithPrimaryTensor:tReceipt secondaryTensor:tCommit name:nil]
                                     toType:MPSDataTypeFloat32 name:@"late"];

    MPSGraphTensorData* cmTD = columnTensor(device, (void*)l_commitdate.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* rcTD = columnTensor(device, (void*)l_receiptdate.data(), N, MPSDataTypeInt32);
    NSDictionary* feeds = @{ tCommit: cmTD, tReceipt: rcTD };

    id<MTLBuffer> lateBuf = allocSharedBuffer(device, N * sizeof(float));
    MPSGraphTensorData* lateTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:lateBuf shape:@[@(N)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:lateTD forKey:lateF];

    // ----------------------------------------------------------------
    // Step 3: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 4: CPU — per-order grouping (contiguous rows, <=7 per order), semi/anti-join logic
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* lateOut = (float*)[lateBuf contents];

    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());
    std::vector<char> statusF((size_t)(max_orderkey + 1), 0);
    for (size_t i = 0; i < o_orderkey.size(); i++)
        if (o_orderstatus[i] == 'F') statusF[(size_t)o_orderkey[i]] = 1;

    int max_suppkey = *std::max_element(s_suppkey.begin(), s_suppkey.end());
    std::vector<int32_t> numwait((size_t)(max_suppkey + 1), 0);

    struct Entry { int suppkey; bool late; };
    std::vector<Entry> group;
    group.reserve(8);

    // Groups are at most 7 rows (TPC-H's LINECNT domain), so plain linear scans over the
    // local buffer beat std::map/std::set — those cost a heap allocation per order otherwise.
    auto flushGroup = [&](int orderkey) {
        if (group.empty()) return;
        if (orderkey < 0 || orderkey > max_orderkey || !statusF[(size_t)orderkey]) { group.clear(); return; }

        int totalLate = 0;
        int distinctCount = 0;
        for (size_t i = 0; i < group.size(); i++) {
            if (group[i].late) totalLate++;
            bool seen = false;
            for (size_t j = 0; j < i; j++) if (group[j].suppkey == group[i].suppkey) { seen = true; break; }
            if (!seen) distinctCount++;
        }
        if (distinctCount >= 2) {
            for (auto& e : group) {
                if (!e.late) continue;
                int lateCountSameSupplier = 0;
                for (auto& o : group) if (o.late && o.suppkey == e.suppkey) lateCountSameSupplier++;
                if (lateCountSameSupplier == totalLate) {
                    if (e.suppkey >= 0 && e.suppkey <= max_suppkey) numwait[(size_t)e.suppkey]++;
                }
            }
        }
        group.clear();
    };

    int curOrderkey = -1;
    for (size_t i = 0; i < N; i++) {
        int ok = l_orderkey[i];
        if (ok != curOrderkey) {
            flushGroup(curOrderkey);
            curOrderkey = ok;
        }
        group.push_back({l_suppkey[i], lateOut[i] > 0.5f});
    }
    flushGroup(curOrderkey);

    struct ResultRow { std::string name; int cnt; };
    std::vector<ResultRow> rows;
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        if (s_nationkey[i] != target_nation) continue;
        int sk = s_suppkey[i];
        if (sk < 0 || sk > max_suppkey || numwait[(size_t)sk] == 0) continue;
        rows.push_back({trimFixed(s_name.data(), i, 25), numwait[(size_t)sk]});
    }
    std::sort(rows.begin(), rows.end(), [](const ResultRow& a, const ResultRow& b) {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        return a.name < b.name;
    });
    if (rows.size() > 100) rows.resize(100);

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q21 Results (first 15 of %zu, LIMIT 100):\n", rows.size());
    printf("+--------------------+---------+\n");
    printf("| s_name             | numwait |\n");
    printf("+--------------------+---------+\n");
    size_t show = std::min(rows.size(), (size_t)15);
    for (size_t i = 0; i < show; i++)
        printf("| %-18s | %7d |\n", rows[i].name.c_str(), rows[i].cnt);
    printf("+--------------------+---------+\n");

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q21", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
