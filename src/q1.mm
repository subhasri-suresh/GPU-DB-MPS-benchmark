#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q1 — Pricing Summary Report
//
// SELECT l_returnflag, l_linestatus,
//        SUM(l_quantity), SUM(l_extendedprice),
//        SUM(l_extendedprice*(1-l_discount)),
//        SUM(l_extendedprice*(1-l_discount)*(1+l_tax)),
//        AVG(l_quantity), AVG(l_extendedprice), AVG(l_discount), COUNT(*)
// FROM lineitem
// WHERE l_shipdate <= DATE '1998-12-01' - INTERVAL '90' DAY  (= 1998-09-02)
// GROUP BY l_returnflag, l_linestatus
// ORDER BY l_returnflag, l_linestatus;
//
// Group encoding (rf_idx * 2 + ls_idx):
//   (A,F)=0  (A,O)=1  (N,F)=2  (N,O)=3  (R,F)=4  (R,O)=5
//
// Strategy: scatter (MPSGraphScatterModeAdd) causes extreme atomic contention
// when 6M threads write into only 6 bins. Instead, use one reductionSum per
// group per metric — 6 groups × 6 metrics = 36 parallel reductions, each O(log N).
// The 6 scalar [1] outputs per metric are concatenated into a single [6] tensor.

void runQ1(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q1: Pricing Summary Report ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load 7 columns in a single pass
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        { 4, ColType::FLOAT},
        { 5, ColType::FLOAT},
        { 6, ColType::FLOAT},
        { 7, ColType::FLOAT},
        { 8, ColType::CHAR1 },
        { 9, ColType::CHAR1 },
        {10, ColType::DATE  },
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& l_qty      = cols.floats(4);
    auto& l_price    = cols.floats(5);
    auto& l_discount = cols.floats(6);
    auto& l_tax      = cols.floats(7);
    auto& l_rf       = cols.chars(8);
    auto& l_ls       = cols.chars(9);
    auto& l_shipdate = cols.ints(10);
    size_t N = l_shipdate.size();
    printf("  Rows: %zu  (parse: %.1f ms)\n", N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: CPU — group_idx per row
    // ----------------------------------------------------------------
    std::vector<int32_t> groupIdx(N);
    for (size_t i = 0; i < N; i++) {
        int rf = (l_rf[i] == 'A') ? 0 : (l_rf[i] == 'N') ? 1 : 2;
        int ls = (l_ls[i] == 'O') ? 1 : 0;
        groupIdx[i] = rf * 2 + ls;
    }

    // ----------------------------------------------------------------
    // Step 3: Build MPSGraph — per-group reductions
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    MPSGraphTensor* tQty      = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"qty"];
    MPSGraphTensor* tPrice    = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"price"];
    MPSGraphTensor* tDiscount = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"disc"];
    MPSGraphTensor* tTax      = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"tax"];
    MPSGraphTensor* tShipdate = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"shipdate"];
    MPSGraphTensor* tGroupIdx = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"groupIdx"];

    // Date filter
    MPSGraphTensor* cutoff    = [graph constantWithScalar:19980902 dataType:MPSDataTypeInt32];
    MPSGraphTensor* dateMaskF = [graph castTensor:
                                    [graph lessThanOrEqualToWithPrimaryTensor:tShipdate
                                                             secondaryTensor:cutoff name:nil]
                                         toType:MPSDataTypeFloat32 name:nil];

    // Derived columns (computed once, shared across all groups)
    MPSGraphTensor* one       = [graph constantWithScalar:1.0f dataType:MPSDataTypeFloat32];
    MPSGraphTensor* discPrice = [graph multiplicationWithPrimaryTensor:tPrice
                                 secondaryTensor:[graph subtractionWithPrimaryTensor:one
                                                             secondaryTensor:tDiscount name:nil]
                                 name:nil];
    MPSGraphTensor* chargeV   = [graph multiplicationWithPrimaryTensor:discPrice
                                 secondaryTensor:[graph additionWithPrimaryTensor:one
                                                          secondaryTensor:tTax name:nil]
                                 name:nil];

    // Date-masked values (multiply once, then group-mask per loop iteration)
    MPSGraphTensor* dQty    = [graph multiplicationWithPrimaryTensor:tQty      secondaryTensor:dateMaskF name:nil];
    MPSGraphTensor* dPrice  = [graph multiplicationWithPrimaryTensor:tPrice    secondaryTensor:dateMaskF name:nil];
    MPSGraphTensor* dDiscP  = [graph multiplicationWithPrimaryTensor:discPrice  secondaryTensor:dateMaskF name:nil];
    MPSGraphTensor* dCharge = [graph multiplicationWithPrimaryTensor:chargeV    secondaryTensor:dateMaskF name:nil];
    MPSGraphTensor* dDisc   = [graph multiplicationWithPrimaryTensor:tDiscount  secondaryTensor:dateMaskF name:nil];

    // Per-group: equality mask → reductionSum for each metric
    // Collect 6 scalar [1] tensors per metric, then concat → [6]
    NSMutableArray* gQty=[@[] mutableCopy], *gPrice=[@[] mutableCopy],
                   *gDiscP=[@[] mutableCopy], *gCharge=[@[] mutableCopy],
                   *gDisc=[@[] mutableCopy],  *gCnt=[@[] mutableCopy];

    for (int g = 0; g < 6; g++) {
        MPSGraphTensor* gConst = [graph constantWithScalar:g dataType:MPSDataTypeInt32];
        MPSGraphTensor* gMaskF = [graph castTensor:
                                     [graph equalWithPrimaryTensor:tGroupIdx
                                                  secondaryTensor:gConst name:nil]
                                          toType:MPSDataTypeFloat32 name:nil];

        // Combined filter: date mask already applied to dXxx; group mask applied here
        [gQty    addObject:[graph reductionSumWithTensor:
                                [graph multiplicationWithPrimaryTensor:dQty   secondaryTensor:gMaskF name:nil]
                            axis:0 name:nil]];
        [gPrice  addObject:[graph reductionSumWithTensor:
                                [graph multiplicationWithPrimaryTensor:dPrice secondaryTensor:gMaskF name:nil]
                            axis:0 name:nil]];
        [gDiscP  addObject:[graph reductionSumWithTensor:
                                [graph multiplicationWithPrimaryTensor:dDiscP secondaryTensor:gMaskF name:nil]
                            axis:0 name:nil]];
        [gCharge addObject:[graph reductionSumWithTensor:
                                [graph multiplicationWithPrimaryTensor:dCharge secondaryTensor:gMaskF name:nil]
                            axis:0 name:nil]];
        [gDisc   addObject:[graph reductionSumWithTensor:
                                [graph multiplicationWithPrimaryTensor:dDisc  secondaryTensor:gMaskF name:nil]
                            axis:0 name:nil]];
        // count: rows passing date filter AND in this group
        [gCnt    addObject:[graph reductionSumWithTensor:
                                [graph multiplicationWithPrimaryTensor:dateMaskF secondaryTensor:gMaskF name:nil]
                            axis:0 name:nil]];
    }

    // Concatenate 6 scalar [1] tensors → [6] output per metric
    MPSGraphTensor* outQty    = [graph concatTensors:gQty    dimension:0 name:@"sumQty"];
    MPSGraphTensor* outPrice  = [graph concatTensors:gPrice  dimension:0 name:@"sumPrice"];
    MPSGraphTensor* outDiscP  = [graph concatTensors:gDiscP  dimension:0 name:@"sumDiscP"];
    MPSGraphTensor* outCharge = [graph concatTensors:gCharge dimension:0 name:@"sumCharge"];
    MPSGraphTensor* outDisc   = [graph concatTensors:gDisc   dimension:0 name:@"sumDisc"];
    MPSGraphTensor* outCnt    = [graph concatTensors:gCnt    dimension:0 name:@"count"];

    // ----------------------------------------------------------------
    // Step 4: Feeds + outputs
    // ----------------------------------------------------------------
    MPSGraphTensorData* qtTD = columnTensor(device, (void*)l_qty.data(),      N, MPSDataTypeFloat32);
    MPSGraphTensorData* prTD = columnTensor(device, (void*)l_price.data(),    N, MPSDataTypeFloat32);
    MPSGraphTensorData* diTD = columnTensor(device, (void*)l_discount.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* txTD = columnTensor(device, (void*)l_tax.data(),      N, MPSDataTypeFloat32);
    MPSGraphTensorData* sdTD = columnTensor(device, (void*)l_shipdate.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* giTD = columnTensor(device, groupIdx.data(),          N, MPSDataTypeInt32);

    NSDictionary* feeds = @{
        tQty:      qtTD, tPrice:    prTD, tDiscount: diTD,
        tTax:      txTD, tShipdate: sdTD, tGroupIdx: giTD
    };

    auto makeOutBuf = [&](MPSGraphTensor* t) -> std::pair<id<MTLBuffer>, MPSGraphTensorData*> {
        id<MTLBuffer> buf = allocSharedBuffer(device, 6 * sizeof(float));
        MPSGraphTensorData* td = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:buf shape:@[@6] dataType:MPSDataTypeFloat32];
        return {buf, td};
    };
    auto [bQty,    tdQty]    = makeOutBuf(outQty);
    auto [bPrice,  tdPrice]  = makeOutBuf(outPrice);
    auto [bDiscP,  tdDiscP]  = makeOutBuf(outDiscP);
    auto [bCharge, tdCharge] = makeOutBuf(outCharge);
    auto [bDisc,   tdDisc]   = makeOutBuf(outDisc);
    auto [bCnt,    tdCnt]    = makeOutBuf(outCnt);

    NSMutableDictionary* results = [NSMutableDictionary dictionary];
    results[outQty]    = tdQty;
    results[outPrice]  = tdPrice;
    results[outDiscP]  = tdDiscP;
    results[outCharge] = tdCharge;
    results[outDisc]   = tdDisc;
    results[outCnt]    = tdCnt;

    // ----------------------------------------------------------------
    // Step 5: 2 warmup + 1 measured
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 6: CPU post — averages and print
    // ----------------------------------------------------------------
    auto tp0 = std::chrono::high_resolution_clock::now();
    float* rQty    = (float*)[bQty    contents];
    float* rPrice  = (float*)[bPrice  contents];
    float* rDiscP  = (float*)[bDiscP  contents];
    float* rCharge = (float*)[bCharge contents];
    float* rDisc   = (float*)[bDisc   contents];
    float* rCount  = (float*)[bCnt    contents];

    static const char* rfChars[] = {"A","A","N","N","R","R"};
    static const char* lsChars[] = {"F","O","F","O","F","O"};

    printf("\n+----------+----------+------------+----------------+----------------+");
    printf("----------------+------------+------------+------------+----------+\n");
    printf("| l_return | l_linest |    sum_qty | sum_base_price | sum_disc_price |");
    printf("     sum_charge |    avg_qty |  avg_price |   avg_disc | count    |\n");
    printf("+----------+----------+------------+----------------+----------------+");
    printf("----------------+------------+------------+------------+----------+\n");
    for (int b = 0; b < 6; b++) {
        if (rCount[b] < 1.0f) continue;
        printf("| %8s | %8s | %10.2f | %14.2f | %14.2f | %14.2f | %10.4f | %10.2f | %10.6f | %8.0f |\n",
               rfChars[b], lsChars[b],
               rQty[b], rPrice[b], rDiscP[b], rCharge[b],
               rQty[b] / rCount[b], rPrice[b] / rCount[b], rDisc[b] / rCount[b],
               rCount[b]);
    }
    printf("+----------+----------+------------+----------------+----------------+");
    printf("----------------+------------+------------+------------+----------+\n");

    auto tp1 = std::chrono::high_resolution_clock::now();
    double postMs = std::chrono::duration<double, std::milli>(tp1 - tp0).count();

    printf("\n  Rows: %zu\n", N);
    printTimingSummary(parseMs, gpuMs, postMs);

    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q1", gpuMs, postMs,
                   std::string(device.name.UTF8String), memGB);
}
