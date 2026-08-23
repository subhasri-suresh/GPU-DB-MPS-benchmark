#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q2 — Minimum Cost Supplier
//
// SELECT s_acctbal, s_name, n_name, p_partkey, p_mfgr, s_address, s_phone, s_comment
// FROM part, supplier, partsupp, nation, region
// WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
//   AND p_size = 15 AND p_type LIKE '%BRASS'
//   AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE'
//   AND ps_supplycost = (
//       SELECT MIN(ps_supplycost) FROM partsupp, supplier, nation, region
//       WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey
//         AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey AND r_name = 'EUROPE')
// ORDER BY s_acctbal DESC, n_name, s_name, p_partkey LIMIT 100;
//
// Strategy (fully GPU except the string filters and final top-N sort):
//   Exploits the same fixed 4-contiguous-partsupp-rows-per-part layout as Q9: for each
//   part, gather its 4 (ps_suppkey, ps_supplycost) candidates, mask out any supplier not
//   in Europe, and take the min supplycost + the suppkey that achieves it, directly index
//   by index over the (small) part table — no scatter/hash join needed.
//   CPU post: reuses the shared postProcessQ2 helper for the ORDER BY/LIMIT 100 + print.

void runQ2(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q2: Minimum Cost Supplier ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto pCols = loadColumnsMulti(g_dataset_path + "part.tbl", {
        {0, ColType::INT        },
        {2, ColType::CHAR_FIXED, 25},   // p_mfgr
        {4, ColType::CHAR_FIXED, 25},   // p_type
        {5, ColType::INT        },      // p_size
    });
    auto& p_partkey = pCols.ints(0);
    auto& p_mfgr    = pCols.chars(2);
    auto& p_type    = pCols.chars(4);
    auto& p_size    = pCols.ints(5);
    size_t partN = p_partkey.size();

    auto psCols = loadColumnsMulti(g_dataset_path + "partsupp.tbl", {
        {1, ColType::INT  },   // ps_suppkey
        {3, ColType::FLOAT},   // ps_supplycost
    });
    auto& ps_suppkey    = psCols.ints(1);
    auto& ps_supplycost = psCols.floats(3);

    auto sCols = loadColumnsMulti(g_dataset_path + "supplier.tbl", {
        {0, ColType::INT        },
        {1, ColType::CHAR_FIXED, 25},   // s_name
        {2, ColType::CHAR_FIXED, 40},   // s_address
        {3, ColType::INT        },      // s_nationkey
        {4, ColType::CHAR_FIXED, 15},   // s_phone
        {5, ColType::FLOAT      },      // s_acctbal
        {6, ColType::CHAR_FIXED, 101},  // s_comment
    });
    auto& s_suppkey   = sCols.ints(0);
    auto& s_name      = sCols.chars(1);
    auto& s_address   = sCols.chars(2);
    auto& s_nationkey = sCols.ints(3);
    auto& s_phone     = sCols.chars(4);
    auto& s_acctbal   = sCols.floats(5);
    auto& s_comment   = sCols.chars(6);

    auto nat = loadNation(g_dataset_path, /*with_regionkey=*/true);
    auto nation_names = buildNationNames(nat.nationkey, nat.name.data(), NationData::NAME_WIDTH);
    auto reg = loadRegion(g_dataset_path);
    int europe_regionkey = -1;
    for (size_t i = 0; i < reg.regionkey.size(); i++)
        if (trimFixed(reg.name.data(), i, RegionData::NAME_WIDTH) == "EUROPE") europe_regionkey = reg.regionkey[i];
    std::vector<int> europe_nations = filterNationsByRegion(nat.nationkey, nat.regionkey, europe_regionkey);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Part: %zu  PartSupp: %zu  Supplier: %zu  (parse: %.1f ms)\n",
           partN, ps_suppkey.size(), s_suppkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — supplier-in-Europe dense map + index (suppkey -> row)
    // ----------------------------------------------------------------
    auto suppBitmap = buildSuppBitmapAndIndex(s_suppkey.data(), s_nationkey.data(), s_suppkey.size(), europe_nations);
    int max_suppkey = 0;
    for (int sk : s_suppkey) max_suppkey = std::max(max_suppkey, sk);
    std::vector<float> supp_in_europe((size_t)(max_suppkey + 1), 0.0f);
    for (size_t i = 0; i < s_suppkey.size(); i++) {
        bool inEurope = false;
        for (int nk : europe_nations) if (s_nationkey[i] == nk) { inEurope = true; break; }
        if (inEurope) supp_in_europe[(size_t)s_suppkey[i]] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: CPU — per-part filter mask (p_size==15 AND p_type LIKE '%BRASS')
    // ----------------------------------------------------------------
    std::vector<float> part_filter(partN, 0.0f);
    for (size_t i = 0; i < partN; i++) {
        std::string type = trimFixed(p_type.data(), i, 25);
        bool brassSuffix = type.size() >= 5 && type.compare(type.size() - 5, 5, "BRASS") == 0;
        if (p_size[i] == 15 && brassSuffix) part_filter[i] = 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 4: Build MPSGraph — per part: min supplycost among Europe suppliers, + which supplier
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tPartFilter = [graph placeholderWithShape:@[@(partN)] dataType:MPSDataTypeFloat32 name:@"part_filter"];
    MPSGraphTensor* tPsSuppkey  = [graph placeholderWithShape:@[@(ps_suppkey.size())] dataType:MPSDataTypeInt32   name:@"ps_suppkey"];
    MPSGraphTensor* tPsCost     = [graph placeholderWithShape:@[@(ps_suppkey.size())] dataType:MPSDataTypeFloat32 name:@"ps_cost"];
    MPSGraphTensor* tSuppEurope = [graph placeholderWithShape:@[@(max_suppkey + 1)] dataType:MPSDataTypeFloat32 name:@"supp_in_europe"];

    // Part-index tensor: 0..partN-1 (used to compute each part's 4-row base into partsupp)
    std::vector<int32_t> partIdxSeq(partN);
    for (size_t i = 0; i < partN; i++) partIdxSeq[i] = (int32_t)i;
    MPSGraphTensor* tPartIdx = [graph placeholderWithShape:@[@(partN)] dataType:MPSDataTypeInt32 name:@"part_idx"];

    MPSGraphTensor* fourI = [graph constantWithScalar:4 dataType:MPSDataTypeInt32];
    MPSGraphTensor* psBase = [graph multiplicationWithPrimaryTensor:tPartIdx secondaryTensor:fourI name:@"ps_base"];

    const float BIG = 1e18f;
    MPSGraphTensor* bigConst = [graph constantWithScalar:BIG shape:@[@(partN)] dataType:MPSDataTypeFloat32];
    MPSGraphTensor* minCost = bigConst;
    MPSGraphTensor* minSupp = [graph constantWithScalar:-1 shape:@[@(partN)] dataType:MPSDataTypeInt32];

    for (int j = 0; j < 4; j++) {
        MPSGraphTensor* jI  = [graph constantWithScalar:j dataType:MPSDataTypeInt32];
        MPSGraphTensor* idx = [graph additionWithPrimaryTensor:psBase secondaryTensor:jI name:nil];
        MPSGraphTensor* candSuppkey = [graph gatherWithUpdatesTensor:tPsSuppkey indicesTensor:idx
                                                                  axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* candCost    = [graph gatherWithUpdatesTensor:tPsCost    indicesTensor:idx
                                                                  axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* candEurope  = [graph gatherWithUpdatesTensor:tSuppEurope indicesTensor:candSuppkey
                                                                  axis:0 batchDimensions:0 name:nil];
        MPSGraphTensor* eligible = [graph greaterThanWithPrimaryTensor:candEurope
                                                        secondaryTensor:[graph constantWithScalar:0.5f
                                                                             dataType:MPSDataTypeFloat32] name:nil];
        MPSGraphTensor* candCostMasked = [graph selectWithPredicateTensor:eligible
                                                        truePredicateTensor:candCost
                                                       falsePredicateTensor:bigConst
                                                                       name:nil];
        MPSGraphTensor* isNewMin = [graph lessThanWithPrimaryTensor:candCostMasked secondaryTensor:minCost name:nil];
        minSupp = [graph selectWithPredicateTensor:isNewMin
                                truePredicateTensor:candSuppkey
                               falsePredicateTensor:minSupp name:nil];
        minCost = [graph minimumWithPrimaryTensor:minCost secondaryTensor:candCostMasked name:nil];
    }

    MPSGraphTensor* found = [graph lessThanWithPrimaryTensor:minCost
                                             secondaryTensor:[graph constantWithScalar:(BIG * 0.5)
                                                                  shape:@[@(partN)] dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* passesFilter = [graph greaterThanWithPrimaryTensor:tPartFilter
                                                        secondaryTensor:[graph constantWithScalar:0.5f
                                                                             dataType:MPSDataTypeFloat32] name:nil];
    MPSGraphTensor* valid = [graph logicalANDWithPrimaryTensor:found secondaryTensor:passesFilter name:@"valid"];

    // Encode: output suppkey if valid else -1 (CPU compacts on this sentinel).
    MPSGraphTensor* outSupp = [graph selectWithPredicateTensor:valid
                                            truePredicateTensor:minSupp
                                           falsePredicateTensor:[graph constantWithScalar:-1 shape:@[@(partN)]
                                                                      dataType:MPSDataTypeInt32]
                                                            name:@"out_supp"];
    MPSGraphTensor* outCost = minCost;

    // ----------------------------------------------------------------
    // Step 5: Feeds + output buffers
    // ----------------------------------------------------------------
    MPSGraphTensorData* pfTD  = columnTensor(device, part_filter.data(), partN, MPSDataTypeFloat32);
    MPSGraphTensorData* pskTD = columnTensor(device, (void*)ps_suppkey.data(),    ps_suppkey.size(), MPSDataTypeInt32);
    MPSGraphTensorData* pscTD = columnTensor(device, (void*)ps_supplycost.data(), ps_suppkey.size(), MPSDataTypeFloat32);
    MPSGraphTensorData* seTD  = columnTensor(device, supp_in_europe.data(), (size_t)(max_suppkey + 1), MPSDataTypeFloat32);
    MPSGraphTensorData* piTD  = columnTensor(device, partIdxSeq.data(), partN, MPSDataTypeInt32);

    NSDictionary* feeds = @{
        tPartFilter: pfTD, tPsSuppkey: pskTD, tPsCost: pscTD, tSuppEurope: seTD, tPartIdx: piTD
    };

    id<MTLBuffer> suppBuf = allocSharedBuffer(device, partN * sizeof(int32_t));
    id<MTLBuffer> costBuf = allocSharedBuffer(device, partN * sizeof(float));
    MPSGraphTensorData* suppTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:suppBuf shape:@[@(partN)] dataType:MPSDataTypeInt32];
    MPSGraphTensorData* costTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:costBuf shape:@[@(partN)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[outSupp] = suppTD;
    results[outCost] = costTD;

    // ----------------------------------------------------------------
    // Step 6: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 7: CPU post — compact valid rows, join + top-N via postProcessQ2
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    int32_t* outSuppData = (int32_t*)[suppBuf contents];
    float*   outCostData = (float*)[costBuf contents];

    std::vector<Q2MatchResult_CPU> gpu_results;
    for (size_t i = 0; i < partN; i++) {
        if (outSuppData[i] < 0) continue;
        Q2MatchResult_CPU r;
        r.partkey = p_partkey[i];
        r.suppkey = outSuppData[i];
        r.supplycost_cents = (unsigned int)(outCostData[i] * 100.0f + 0.5f);
        gpu_results.push_back(r);
    }

    postProcessQ2(gpu_results.data(), (uint)gpu_results.size(), suppBitmap.index,
                  s_acctbal.data(), s_nationkey.data(), s_name.data(), s_address.data(),
                  s_phone.data(), s_comment.data(), nation_names,
                  p_partkey.data(), partN, p_mfgr.data());

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Parts: %zu\n", partN);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q2", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
