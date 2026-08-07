#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q13 — Customer Distribution
//
// SELECT c_count, COUNT(*) AS custdist
// FROM (
//     SELECT c_custkey, COUNT(o_orderkey) AS c_count
//     FROM customer LEFT OUTER JOIN orders
//         ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'
//     GROUP BY c_custkey
// ) AS c_orders
// GROUP BY c_count
// ORDER BY custdist DESC, c_count DESC
//
// Strategy (mirrors the raw-Metal baseline's pattern-match + count + histogram kernels):
//   CPU:       o_qualifies[i] = 1.0 if o_comment does NOT contain "special" followed
//              later by "requests" (MPSGraph has no string ops, so — like Q10/Q12's
//              char-column masks — the substring search is precomputed on the CPU).
//   MPSGraph:  scatter-add the qualifies mask directly into a dense per-customer array
//              indexed by o_custkey — since the mask is already 0/1, the scatter *is*
//              the COUNT(o_orderkey) aggregate, in one pass over orders.
//   CPU post:  bucket each customer's count into a 256-bin histogram (customers with
//              zero qualifying orders included via the LEFT OUTER JOIN semantics — the
//              per-customer array defaults to 0), then reuse the shared postProcessQ13
//              helper for the sort + print.

void runQ13(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q13: Customer Distribution ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load tables
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();

    auto oCols = loadColumnsMulti(g_dataset_path + "orders.tbl", {
        {1, ColType::INT}, {8, ColType::CHAR_FIXED, 80},
    });
    auto& o_custkey = oCols.ints(1);
    auto& o_comment = oCols.chars(8);
    size_t M = o_custkey.size();

    auto cCols = loadColumnsMulti(g_dataset_path + "customer.tbl", {{0, ColType::INT}});
    auto& c_custkey = cCols.ints(0);

    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("  Orders: %zu  Customer: %zu  (parse: %.1f ms)\n", M, c_custkey.size(), parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — pattern match: NOT LIKE '%special%requests%'
    // ----------------------------------------------------------------
    std::vector<float> qualifies(M);
    for (size_t i = 0; i < M; i++) {
        std::string comment = trimFixed(o_comment.data(), i, 80);
        size_t specialPos = comment.find("special");
        bool matched = false;
        if (specialPos != std::string::npos)
            matched = comment.find("requests", specialPos + 7) != std::string::npos;
        qualifies[i] = matched ? 0.0f : 1.0f;
    }

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph
    // ----------------------------------------------------------------
    int max_custkey = *std::max_element(c_custkey.begin(), c_custkey.end());

    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tOrdCustkey = [graph placeholderWithShape:@[@(M)]
                                                     dataType:MPSDataTypeInt32   name:@"o_custkey"];
    MPSGraphTensor* tQualifies  = [graph placeholderWithShape:@[@(M)]
                                                     dataType:MPSDataTypeFloat32 name:@"qualifies"];

    // COUNT(o_orderkey) per custkey, filtered by the qualifies mask, via scatter-add
    MPSGraphTensor* zerosCounts = [graph constantWithScalar:0.0 shape:@[@(max_custkey + 1)]
                                                   dataType:MPSDataTypeFloat32];
    MPSGraphTensor* counts = [graph scatterWithDataTensor:zerosCounts
                                            updatesTensor:tQualifies
                                            indicesTensor:tOrdCustkey
                                                     axis:0
                                                     mode:MPSGraphScatterModeAdd
                                                     name:@"counts"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + output buffer
    // ----------------------------------------------------------------
    MPSGraphTensorData* ckTD = columnTensor(device, (void*)o_custkey.data(), M, MPSDataTypeInt32);
    MPSGraphTensorData* qTD  = columnTensor(device, qualifies.data(),        M, MPSDataTypeFloat32);

    NSDictionary* feeds = @{ tOrdCustkey: ckTD, tQualifies: qTD };

    id<MTLBuffer> countsBuf = allocSharedBuffer(device, (size_t)(max_custkey + 1) * sizeof(float));
    MPSGraphTensorData* countsTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:countsBuf shape:@[@(max_custkey + 1)] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[counts] = countsTD;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — bucket per-customer counts into a histogram, print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* counts_map = (float*)[countsBuf contents];

    const uint HIST_MAX_BINS = 256;
    std::vector<uint> histogram(HIST_MAX_BINS, 0);
    for (size_t i = 0; i < c_custkey.size(); i++) {
        int ck = c_custkey[i];
        if (ck < 0 || ck > max_custkey) continue;
        uint count = (uint)(counts_map[ck] + 0.5f);   // exact small integers; round guards fp noise
        if (count < HIST_MAX_BINS) histogram[count]++;
    }

    postProcessQ13(histogram.data(), HIST_MAX_BINS);

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Orders: %zu  Customers: %zu\n", M, c_custkey.size());
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q13", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
