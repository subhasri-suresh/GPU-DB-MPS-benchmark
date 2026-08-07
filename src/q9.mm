#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q9 — Product Type Profit Measure
//
// SELECT n_name, EXTRACT(year FROM o_orderdate) AS o_year,
//        SUM(l_extendedprice*(1-l_discount) - ps_supplycost*l_quantity) AS sum_profit
// FROM part, supplier, lineitem, partsupp, orders, nation
// WHERE s_suppkey  = l_suppkey
//   AND ps_suppkey = l_suppkey  AND ps_partkey = l_partkey
//   AND p_partkey  = l_partkey
//   AND o_orderkey = l_orderkey
//   AND s_nationkey = n_nationkey
//   AND p_name LIKE '%green%'
// GROUP BY n_name, o_year ORDER BY n_name, o_year DESC
//
// Strategy (fully GPU except the three small dense-array builds below):
//   CPU build: part bitmap (contains 'green'), supplier-nation map, orders-year map.
//   MPSGraph:  gather part/supp/order lookups; resolve the composite (partkey,suppkey)
//              -> supplycost join on GPU by exploiting TPC-H's fixed PARTSUPP layout
//              (exactly 4 contiguous rows per part, sorted by partkey): gather the 4
//              candidate rows at base=(partkey-1)*4 and mask-select the one whose
//              suppkey matches. Compute profit per row, scatter into (nation × year)
//              grid of 25×10 = 250 buckets.
//   CPU post:  decode buckets, sort by (n_name ASC, year DESC), print.

static constexpr int MAX_YEARS = 10;
static constexpr int MIN_YEAR  = 1990;
static constexpr int NUM_BUCKETS = 25 * MAX_YEARS;  // 250

void runQ9(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q9: Product Type Profit Measure ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    // part: p_partkey (col 0), p_name (col 1, 55 chars)
    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT        },
        {1, ColType::CHAR_FIXED, 55},
    });
    auto& p_partkey = pCols.ints(0);
    auto& p_name    = pCols.chars(1);

    auto sup = loadSupplierBasic(g_dataset_path);
    auto& s_suppkey   = sup.suppkey;
    auto& s_nationkey = sup.nationkey;

    // ps_partkey is not loaded: it's implicit from row position. TPC-H guarantees
    // partsupp has exactly 4 contiguous rows per part, sorted by partkey, so row
    // (partkey-1)*4 + j (j=0..3) is that part's j-th supplier entry.
    auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl", {
        {1, ColType::INT  },   // ps_suppkey
        {3, ColType::FLOAT},   // ps_supplycost
    });
    auto& ps_suppkey    = psCols.ints(1);
    auto& ps_supplycost = psCols.floats(3);
    size_t psN = ps_suppkey.size();

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {0, ColType::INT },   // o_orderkey
        {4, ColType::DATE},   // o_orderdate
    });
    auto& o_orderkey  = oCols.ints(0);
    auto& o_orderdate = oCols.ints(4);

    auto lCols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 0, ColType::INT  },   // l_orderkey
        { 1, ColType::INT  },   // l_partkey
        { 2, ColType::INT  },   // l_suppkey
        { 4, ColType::FLOAT},   // l_quantity
        { 5, ColType::FLOAT},   // l_extendedprice
        { 6, ColType::FLOAT},   // l_discount
    });
    auto& l_orderkey = lCols.ints(0);
    auto& l_partkey  = lCols.ints(1);
    auto& l_suppkey  = lCols.ints(2);
    auto& l_qty      = lCols.floats(4);
    auto& l_extprice = lCols.floats(5);
    auto& l_discount = lCols.floats(6);
    size_t N = l_orderkey.size();

    auto nat = loadNation(g_dataset_path);
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  Supplier: %zu  PartSupp: %zu  Orders: %zu  Lineitem: %zu  (parse: %.1f ms)\n",
           p_partkey.size(), s_suppkey.size(), psN, o_orderkey.size(), N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — part bitmap (1.0 if name contains 'green', else 0.0)
    // ----------------------------------------------------------------
    int max_partkey = *std::max_element(p_partkey.begin(), p_partkey.end());
    std::vector<float> part_green((size_t)(max_partkey + 1), 0.0f);
    for (size_t i = 0; i < p_partkey.size(); i++) {
        const char* nm = p_name.data() + i * 55;
        for (int j = 0; j <= 49; j++) {
            if (memcmp(nm + j, "green", 5) == 0) {
                part_green[(size_t)p_partkey[i]] = 1.0f;
                break;
            }
        }
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — supplier-nation map (suppkey → nationkey, or -1)
    // ----------------------------------------------------------------
    int max_suppkey = *std::max_element(s_suppkey.begin(), s_suppkey.end());
    std::vector<int32_t> supp_nat_map((size_t)(max_suppkey + 1), -1);
    for (size_t i = 0; i < s_suppkey.size(); i++)
        supp_nat_map[(size_t)s_suppkey[i]] = s_nationkey[i];

    // ----------------------------------------------------------------
    // Step 4: CPU — orders year map (orderkey → year, or -1)
    // ----------------------------------------------------------------
    int max_orderkey = *std::max_element(o_orderkey.begin(), o_orderkey.end());
    std::vector<int32_t> orders_year_map((size_t)(max_orderkey + 1), -1);
    for (size_t i = 0; i < o_orderkey.size(); i++)
        orders_year_map[(size_t)o_orderkey[i]] = o_orderdate[i] / 10000;

    // ----------------------------------------------------------------
    // Step 5: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tPartGreen   = [graph placeholderWithShape:@[@(max_partkey + 1)]
                                                      dataType:MPSDataTypeFloat32 name:@"part_green"];
    MPSGraphTensor* tSuppNatMap  = [graph placeholderWithShape:@[@(max_suppkey + 1)]
                                                      dataType:MPSDataTypeInt32   name:@"supp_nat_map"];
    MPSGraphTensor* tOrdYearMap  = [graph placeholderWithShape:@[@(max_orderkey + 1)]
                                                      dataType:MPSDataTypeInt32   name:@"ord_year_map"];
    MPSGraphTensor* tLPartkey    = [graph placeholderWithShape:@[@(N)]
                                                      dataType:MPSDataTypeInt32   name:@"l_partkey"];
    MPSGraphTensor* tLSuppkey    = [graph placeholderWithShape:@[@(N)]
                                                      dataType:MPSDataTypeInt32   name:@"l_suppkey"];
    MPSGraphTensor* tLOrderkey   = [graph placeholderWithShape:@[@(N)]
                                                      dataType:MPSDataTypeInt32   name:@"l_orderkey"];
    MPSGraphTensor* tQty         = [graph placeholderWithShape:@[@(N)]
                                                      dataType:MPSDataTypeFloat32 name:@"qty"];
    MPSGraphTensor* tExtprice    = [graph placeholderWithShape:@[@(N)]
                                                      dataType:MPSDataTypeFloat32 name:@"extprice"];
    MPSGraphTensor* tDiscount    = [graph placeholderWithShape:@[@(N)]
                                                      dataType:MPSDataTypeFloat32 name:@"discount"];
    MPSGraphTensor* tPsSuppkey   = [graph placeholderWithShape:@[@(psN)]
                                                      dataType:MPSDataTypeInt32   name:@"ps_suppkey"];
    MPSGraphTensor* tPsCost      = [graph placeholderWithShape:@[@(psN)]
                                                      dataType:MPSDataTypeFloat32 name:@"ps_supplycost"];

    // Gather lookups per lineitem row
    MPSGraphTensor* partMatch  = [graph gatherWithUpdatesTensor:tPartGreen
                                                  indicesTensor:tLPartkey
                                                           axis:0 batchDimensions:0 name:@"part_match"];
    MPSGraphTensor* suppNation = [graph gatherWithUpdatesTensor:tSuppNatMap
                                                  indicesTensor:tLSuppkey
                                                           axis:0 batchDimensions:0 name:@"supp_nat"];
    MPSGraphTensor* orderYear  = [graph gatherWithUpdatesTensor:tOrdYearMap
                                                  indicesTensor:tLOrderkey
                                                           axis:0 batchDimensions:0 name:@"ord_year"];

    // Composite-key (partkey,suppkey) join, resolved entirely on GPU: TPC-H guarantees
    // partsupp has exactly 4 contiguous rows per part, sorted by partkey, so row
    // (partkey-1)*4 + j (j=0..3) holds that part's j-th supplier entry. Gather the 4
    // candidates, compare suppkey, and mask-select the one that matches.
    MPSGraphTensor* oneI  = [graph constantWithScalar:1 dataType:MPSDataTypeInt32];
    MPSGraphTensor* fourI = [graph constantWithScalar:4 dataType:MPSDataTypeInt32];
    MPSGraphTensor* psBase = [graph multiplicationWithPrimaryTensor:
                                  [graph subtractionWithPrimaryTensor:tLPartkey secondaryTensor:oneI name:nil]
                              secondaryTensor:fourI name:@"ps_base"];

    MPSGraphTensor* supplycost = [graph constantWithScalar:0.0 shape:@[@(N)] dataType:MPSDataTypeFloat32];
    for (int j = 0; j < 4; j++) {
        MPSGraphTensor* jI  = [graph constantWithScalar:j dataType:MPSDataTypeInt32];
        MPSGraphTensor* idx = [graph additionWithPrimaryTensor:psBase secondaryTensor:jI name:nil];

        MPSGraphTensor* candSuppkey = [graph gatherWithUpdatesTensor:tPsSuppkey indicesTensor:idx
                                                                  axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* candCost    = [graph gatherWithUpdatesTensor:tPsCost    indicesTensor:idx
                                                                  axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* matchF = [graph castTensor:
                                      [graph equalWithPrimaryTensor:candSuppkey secondaryTensor:tLSuppkey name:nil]
                                  toType:MPSDataTypeFloat32 name:nil];
        MPSGraphTensor* term = [graph multiplicationWithPrimaryTensor:matchF secondaryTensor:candCost name:nil];
        supplycost = [graph additionWithPrimaryTensor:supplycost secondaryTensor:term name:nil];
    }

    // Validity conditions
    MPSGraphTensor* zeroI       = [graph constantWithScalar:0   dataType:MPSDataTypeInt32];
    MPSGraphTensor* zeroF       = [graph constantWithScalar:0.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* halfF       = [graph constantWithScalar:0.5f dataType:MPSDataTypeFloat32];

    // partMatch is already 0.0/1.0 float; threshold at 0.5 to get bool
    MPSGraphTensor* partBool    = [graph greaterThanWithPrimaryTensor:partMatch
                                                     secondaryTensor:halfF name:nil];
    MPSGraphTensor* validSupp   = [graph greaterThanOrEqualToWithPrimaryTensor:suppNation
                                                               secondaryTensor:zeroI name:nil];
    MPSGraphTensor* validYear   = [graph greaterThanOrEqualToWithPrimaryTensor:orderYear
                                                               secondaryTensor:zeroI name:nil];
    // valid partsupp join: supplycost > 0 (TPC-H supplycost is always > 0; a row with
    // no matching (partkey,suppkey) among the 4 candidates leaves the mask-selected
    // sum at exactly 0.0)
    MPSGraphTensor* validPS     = [graph greaterThanWithPrimaryTensor:supplycost
                                                       secondaryTensor:zeroF name:nil];

    // Combined mask (all four conditions) → float
    MPSGraphTensor* maskF = [graph castTensor:
                                 [graph logicalANDWithPrimaryTensor:
                                     [graph logicalANDWithPrimaryTensor:
                                         [graph logicalANDWithPrimaryTensor:partBool
                                                            secondaryTensor:validSupp name:nil]
                                      secondaryTensor:validYear name:nil]
                                  secondaryTensor:validPS name:nil]
                             toType:MPSDataTypeFloat32 name:@"mask"];

    // Profit per row = (extprice*(1-discount) - supplycost*qty) * mask
    MPSGraphTensor* oneF     = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* revenue  = [graph multiplicationWithPrimaryTensor:tExtprice
                                    secondaryTensor:[graph subtractionWithPrimaryTensor:oneF
                                                                        secondaryTensor:tDiscount name:nil]
                                    name:nil];
    MPSGraphTensor* cost     = [graph multiplicationWithPrimaryTensor:supplycost
                                    secondaryTensor:tQty name:nil];
    MPSGraphTensor* profit   = [graph multiplicationWithPrimaryTensor:
                                    [graph subtractionWithPrimaryTensor:revenue secondaryTensor:cost name:nil]
                                secondaryTensor:maskF name:@"profit"];

    // Group key = nation * MAX_YEARS + (year - MIN_YEAR)
    // Clamped to ≥ 0 so invalid rows scatter into bucket 0 (profit = 0, harmless)
    MPSGraphTensor* maxYearsTensor = [graph constantWithScalar:MAX_YEARS dataType:MPSDataTypeInt32];
    MPSGraphTensor* minYearTensor  = [graph constantWithScalar:MIN_YEAR  dataType:MPSDataTypeInt32];
    MPSGraphTensor* groupKey = [graph additionWithPrimaryTensor:
                                    [graph multiplicationWithPrimaryTensor:suppNation
                                                           secondaryTensor:maxYearsTensor name:nil]
                                secondaryTensor:
                                    [graph subtractionWithPrimaryTensor:orderYear
                                                        secondaryTensor:minYearTensor name:nil]
                                name:@"group_key"];
    MPSGraphTensor* clampedKey = [graph maximumWithPrimaryTensor:groupKey
                                                secondaryTensor:zeroI name:@"clamped_key"];

    // Scatter profit into 250-bucket grid
    MPSGraphTensor* zerosGrid   = [graph constantWithScalar:0.0 shape:@[@(NUM_BUCKETS)]
                                                   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* groupProfit = [graph scatterWithDataTensor:zerosGrid
                                                updatesTensor:profit
                                                indicesTensor:clampedKey
                                                         axis:0
                                                         mode:MPSGraphScatterModeAdd
                                                         name:@"group_profit"];

    // ----------------------------------------------------------------
    // Step 6: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* pgTD   = columnTensor(device, part_green.data(),
                                              (size_t)(max_partkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* snTD   = columnTensor(device, supp_nat_map.data(),
                                              (size_t)(max_suppkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* oyTD   = columnTensor(device, orders_year_map.data(),
                                              (size_t)(max_orderkey + 1), MPSDataTypeInt32);
    MPSGraphTensorData* pkTD   = columnTensor(device, (void*)l_partkey.data(),   N, MPSDataTypeInt32);
    MPSGraphTensorData* skTD   = columnTensor(device, (void*)l_suppkey.data(),   N, MPSDataTypeInt32);
    MPSGraphTensorData* okTD   = columnTensor(device, (void*)l_orderkey.data(),  N, MPSDataTypeInt32);
    MPSGraphTensorData* qTD    = columnTensor(device, (void*)l_qty.data(),       N, MPSDataTypeFloat32);
    MPSGraphTensorData* epTD   = columnTensor(device, (void*)l_extprice.data(),  N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD   = columnTensor(device, (void*)l_discount.data(),  N, MPSDataTypeFloat32);
    MPSGraphTensorData* pskTD  = columnTensor(device, (void*)ps_suppkey.data(),    psN, MPSDataTypeInt32);
    MPSGraphTensorData* pscTD  = columnTensor(device, (void*)ps_supplycost.data(), psN, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tPartGreen:  pgTD, tSuppNatMap: snTD, tOrdYearMap: oyTD,
        tLPartkey:   pkTD, tLSuppkey:   skTD, tLOrderkey:  okTD,
        tQty:        qTD,  tExtprice:   epTD, tDiscount:   diTD,
        tPsSuppkey:  pskTD, tPsCost:    pscTD,
    };

    id<MTLBuffer> profitBuf = allocSharedBuffer(device, NUM_BUCKETS * sizeof(float));
    MPSGraphTensorData* profitTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:profitBuf shape:@[@(NUM_BUCKETS)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[groupProfit] = profitTD;

    // ----------------------------------------------------------------
    // Step 7: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 8: CPU post — decode buckets, sort, print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* gp = (float*)[profitBuf contents];

    std::vector<Q9Result> rows;
    for (int nat_idx = 0; nat_idx < 25; nat_idx++) {
        for (int yr = 0; yr < MAX_YEARS; yr++) {
            float v = gp[nat_idx * MAX_YEARS + yr];
            if (v != 0.0f)
                rows.push_back({nat_idx, MIN_YEAR + yr, v});
        }
    }
    std::sort(rows.begin(), rows.end(), [&](const Q9Result& a, const Q9Result& b) {
        std::string na = nation_names.count(a.nationkey) ? nation_names.at(a.nationkey) : "";
        std::string nb = nation_names.count(b.nationkey) ? nation_names.at(b.nationkey) : "";
        if (na != nb) return na < nb;
        return a.year > b.year;
    });

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\nTPC-H Q9 Results (first 15):\n");
    printf("+--------------------+------+----------------+\n");
    printf("| n_name             | year |     sum_profit |\n");
    printf("+--------------------+------+----------------+\n");
    size_t show = std::min(rows.size(), (size_t)15);
    for (size_t i = 0; i < show; i++) {
        std::string nm = nation_names.count(rows[i].nationkey)
                       ? nation_names.at(rows[i].nationkey) : "?";
        printf("| %-18s | %4d | %14.2f |\n", nm.c_str(), rows[i].year, rows[i].profit);
    }
    printf("+--------------------+------+----------------+\n");
    printf("Total (nation,year) groups: %zu\n", rows.size());

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q9", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
