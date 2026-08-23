#import "mps_infra.h"
#import <Foundation/Foundation.h>
#import <set>

// TPC-H Q16 — Parts/Supplier Relationship
//
// SELECT p_brand, p_type, p_size, COUNT(DISTINCT ps_suppkey) AS supplier_cnt
// FROM partsupp, part
// WHERE p_partkey = ps_partkey
//   AND p_brand <> 'Brand#45'
//   AND p_type NOT LIKE 'MEDIUM POLISHED%'
//   AND p_size IN (49, 14, 23, 45, 19, 3, 36, 9)
//   AND ps_suppkey NOT IN (
//       SELECT s_suppkey FROM supplier WHERE s_comment LIKE '%Customer%Complaints%')
// GROUP BY p_brand, p_type, p_size
// ORDER BY supplier_cnt DESC, p_brand, p_type, p_size;
//
// Strategy: unlike the fixed-cardinality group-bys (Q1's 6 groups, Q4's 5), the number
// of (p_brand, p_type, p_size) groups here is data-dependent and COUNT(DISTINCT ...) has
// no MPSGraph primitive (a scatter-add would double-count a supplier appearing in two
// different parts of the same group). So the split is:
//   CPU build: per-part group id (first-seen order, only for parts passing the static
//              filters) and the supplier-complaint bitmap.
//   MPSGraph:  gather both per partsupp row and AND them into a single valid mask —
//              this is the one genuinely large-N pass (partsupp: 800K+ rows).
//   CPU post:  the GPU mask leaves an O(psN)-sized "which (group,supplier) pairs survived"
//              problem, which is fundamentally a distinct-set aggregation — done with a
//              std::set per group, same spirit as Q9/Q13's CPU decode of GPU output.

void runQ16(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q16: Parts/Supplier Relationship ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT        },
        {3, ColType::CHAR_FIXED, 10},   // p_brand
        {4, ColType::CHAR_FIXED, 25},   // p_type
        {5, ColType::INT        },      // p_size
    });
    auto& p_partkey = pCols.ints(0);
    auto& p_brand   = pCols.chars(3);
    auto& p_type    = pCols.chars(4);
    auto& p_size    = pCols.ints(5);

    auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl", {
        {0, ColType::INT},   // ps_partkey
        {1, ColType::INT},   // ps_suppkey
    });
    auto& ps_partkey = psCols.ints(0);
    auto& ps_suppkey = psCols.ints(1);
    size_t psN = ps_partkey.size();

    auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl", {
        {0, ColType::INT        },
        {6, ColType::CHAR_FIXED, 101},  // s_comment
    });
    auto& s_suppkey = sCols.ints(0);
    auto& s_comment = sCols.chars(6);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  PartSupp: %zu  Supplier: %zu  (parse: %.1f ms)\n",
           p_partkey.size(), psN, s_suppkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — assign a group id to each qualifying part; build supplier complaint bitmap
    // ----------------------------------------------------------------
    static const int TARGET_SIZES[] = {49, 14, 23, 45, 19, 3, 36, 9};
    auto sizeOk = [](int sz) {
        for (int s : TARGET_SIZES) if (s == sz) return true;
        return false;
    };

    int max_partkey = *std::max_element(p_partkey.begin(), p_partkey.end());
    std::vector<int32_t> part_group((size_t)(max_partkey + 1), -1);
    struct GroupKey { std::string brand, type; int size; };
    std::vector<GroupKey> groupKeys;
    std::map<std::tuple<std::string, std::string, int>, int> groupMap;

    for (size_t i = 0; i < p_partkey.size(); i++) {
        std::string brand = trimFixed(p_brand.data(), i, 10);
        std::string type  = trimFixed(p_type.data(), i, 25);
        int sz = p_size[i];
        if (brand == "Brand#45") continue;
        if (type.rfind("MEDIUM POLISHED", 0) == 0) continue;
        if (!sizeOk(sz)) continue;

        auto key = std::make_tuple(brand, type, sz);
        auto it = groupMap.find(key);
        int gid;
        if (it == groupMap.end()) {
            gid = (int)groupKeys.size();
            groupMap[key] = gid;
            groupKeys.push_back({brand, type, sz});
        } else {
            gid = it->second;
        }
        part_group[(size_t)p_partkey[i]] = gid;
    }
    int numGroups = (int)groupKeys.size();

    int max_suppkey = *std::max_element(s_suppkey.begin(), s_suppkey.end());
    std::vector<float> supp_complaint((size_t)(max_suppkey + 1), 0.0f);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        std::string comment = trimFixed(s_comment.data(), i, 101);
        size_t custPos = comment.find("Customer");
        bool matched = custPos != std::string::npos &&
                       comment.find("Complaints", custPos + 8) != std::string::npos;
        if (matched) supp_complaint[(size_t)s_suppkey[i]] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph — per-partsupp-row valid mask
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tPsPartkey = [graph placeholderWithShape:@[@(psN)] dataType:MPSDataTypeInt32 name:@"ps_partkey"];
    MPSGraphTensor* tPsSuppkey = [graph placeholderWithShape:@[@(psN)] dataType:MPSDataTypeInt32 name:@"ps_suppkey"];
    MPSGraphTensor* tPartGroup = [graph placeholderWithShape:@[@(max_partkey + 1)] dataType:MPSDataTypeInt32
                                                     name:@"part_group"];
    MPSGraphTensor* tSuppComplaint = [graph placeholderWithShape:@[@(max_suppkey + 1)] dataType:MPSDataTypeFloat32
                                                         name:@"supp_complaint"];

    MPSGraphTensor* rowGroup = [graph gatherWithUpdatesTensor:tPartGroup indicesTensor:tPsPartkey
                                                          axis:0 batchDimensions:0 name:nil];
    MPSGraphTensor* rowComplaint = [graph gatherWithUpdatesTensor:tSuppComplaint indicesTensor:tPsSuppkey
                                                              axis:0 batchDimensions:0 name:nil];

    MPSGraphTensor* negOne = [graph constantWithScalar:-1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* hasGroup = [graph notEqualWithPrimaryTensor:rowGroup secondaryTensor:negOne name:nil];
    MPSGraphTensor* notComplaint = [graph lessThanWithPrimaryTensor:rowComplaint
                                                     secondaryTensor:[graph constantWithScalar:0.5f
                                                                          dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* validMask = [graph logicalANDWithPrimaryTensor:hasGroup secondaryTensor:notComplaint name:nil];

    // Encode: keyed_group = rowGroup if valid, else -1 (dumped by CPU decode below).
    MPSGraphTensor* keyedGroup = [graph selectWithPredicateTensor:validMask
                                                truePredicateTensor:rowGroup
                                               falsePredicateTensor:[graph constantWithScalar:-1 shape:@[@(psN)]
                                                                          dataType:MPSDataTypeInt32]
                                                               name:@"keyed_group"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* ppkTD = columnTensor(device, (void*)ps_partkey.data(), psN, MPSDataTypeInt32);
    MPSGraphTensorData* pskTD = columnTensor(device, (void*)ps_suppkey.data(), psN, MPSDataTypeInt32);
    MPSGraphTensorData* pgTD  = columnTensor(device, part_group.data(), (size_t)(max_partkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* scTD  = columnTensor(device, supp_complaint.data(), (size_t)(max_suppkey + 1), MPSDataTypeFloat32);

    NSDictionary* feeds = @{ tPsPartkey: ppkTD, tPsSuppkey: pskTD, tPartGroup: pgTD, tSuppComplaint: scTD };

    id<MTLBuffer> keyedGroupBuf = allocSharedBuffer(device, psN * sizeof(int32_t));
    MPSGraphTensorData* keyedGroupTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:keyedGroupBuf shape:@[@(psN)] dataType:MPSDataTypeInt32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[keyedGroup] = keyedGroupTD;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — distinct supplier count per group, sort, print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    int32_t* keyedGroupOut = (int32_t*)[keyedGroupBuf contents];

    std::vector<std::set<int>> groupSuppliers(numGroups);
    for (size_t i = 0; i < psN; i++) {
        int g = keyedGroupOut[i];
        if (g >= 0) groupSuppliers[g].insert(ps_suppkey[i]);
    }

    struct ResultRow { std::string brand, type; int size; int cnt; };
    std::vector<ResultRow> results_rows;
    results_rows.reserve(numGroups);
    for (int g = 0; g < numGroups; g++)
        results_rows.push_back({groupKeys[g].brand, groupKeys[g].type, groupKeys[g].size,
                                 (int)groupSuppliers[g].size()});

    std::sort(results_rows.begin(), results_rows.end(), [](const ResultRow& a, const ResultRow& b) {
        if (a.cnt != b.cnt) return a.cnt > b.cnt;
        if (a.brand != b.brand) return a.brand < b.brand;
        if (a.type != b.type) return a.type < b.type;
        return a.size < b.size;
    });

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q16 Results (first 15 of %d groups):\n", numGroups);
    printf("+------------+---------------------------+------+---------------+\n");
    printf("| p_brand    | p_type                    | size | supplier_cnt  |\n");
    printf("+------------+---------------------------+------+---------------+\n");
    size_t show = std::min(results_rows.size(), (size_t)15);
    for (size_t i = 0; i < show; i++)
        printf("| %-10s | %-25s | %4d | %13d |\n", results_rows[i].brand.c_str(),
               results_rows[i].type.c_str(), results_rows[i].size, results_rows[i].cnt);
    printf("+------------+---------------------------+------+---------------+\n");

    printf("\n  PartSupp rows: %zu\n", psN);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q16", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
