#import "mps_infra.h"
#import <Foundation/Foundation.h>

// TPC-H Q6 — Forecasting Revenue Change
//
// SELECT SUM(l_extendedprice * l_discount) AS revenue
// FROM lineitem
// WHERE l_shipdate >= '1994-01-01'
//   AND l_shipdate <  '1995-01-01'
//   AND l_discount BETWEEN 0.05 AND 0.07
//   AND l_quantity < 24;
//
// Expected SF-1: $123,141,078.23

void runQ6(id<MTLDevice> device, id<MTLCommandQueue> queue) {
    printf("\n--- TPC-H Q6: Forecasting Revenue Change ---\n");

    // ----------------------------------------------------------------
    // Step 1: Load 4 columns from lineitem.tbl in a single pass
    // ----------------------------------------------------------------
    auto t0 = std::chrono::high_resolution_clock::now();
    auto cols = loadColumnsMulti(g_dataset_path + "lineitem.tbl", {
        {10, ColType::DATE },   // l_shipdate       → YYYYMMDD int
        { 6, ColType::FLOAT},   // l_discount
        { 4, ColType::FLOAT},   // l_quantity
        { 5, ColType::FLOAT},   // l_extendedprice
    });
    auto t1 = std::chrono::high_resolution_clock::now();
    double parseMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    auto& shipdate = cols.ints(10);
    auto& discount = cols.floats(6);
    auto& quantity = cols.floats(4);
    auto& price    = cols.floats(5);
    size_t N = shipdate.size();
    printf("  Rows: %zu  (parse: %.1f ms)\n", N, parseMs);

    // ----------------------------------------------------------------
    // Step 2: Build MPSGraph
    // ----------------------------------------------------------------
    MPSGraph* graph = [[MPSGraph alloc] init];

    // Input placeholders — one per column
    MPSGraphTensor* tShipdate = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeInt32   name:@"shipdate"];
    MPSGraphTensor* tDiscount = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"discount"];
    MPSGraphTensor* tQuantity = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"quantity"];
    MPSGraphTensor* tPrice    = [graph placeholderWithShape:@[@(N)] dataType:MPSDataTypeFloat32 name:@"price"];

    // Scalar constants for filter thresholds
    MPSGraphTensor* startDate   = [graph constantWithScalar:19940101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* endDate     = [graph constantWithScalar:19950101 dataType:MPSDataTypeInt32];
    MPSGraphTensor* minDiscount = [graph constantWithScalar:0.05f    dataType:MPSDataTypeFloat32];
    MPSGraphTensor* maxDiscount = [graph constantWithScalar:0.07f    dataType:MPSDataTypeFloat32];
    MPSGraphTensor* maxQty      = [graph constantWithScalar:24.0f    dataType:MPSDataTypeFloat32];

    // 5 filter conditions — each produces a bool tensor of shape [N]
    MPSGraphTensor* c1 = [graph greaterThanOrEqualToWithPrimaryTensor:tShipdate
                                                     secondaryTensor:startDate   name:nil];
    MPSGraphTensor* c2 = [graph lessThanWithPrimaryTensor:tShipdate
                                          secondaryTensor:endDate                name:nil];
    MPSGraphTensor* c3 = [graph greaterThanOrEqualToWithPrimaryTensor:tDiscount
                                                     secondaryTensor:minDiscount name:nil];
    MPSGraphTensor* c4 = [graph lessThanOrEqualToWithPrimaryTensor:tDiscount
                                                   secondaryTensor:maxDiscount   name:nil];
    MPSGraphTensor* c5 = [graph lessThanWithPrimaryTensor:tQuantity
                                          secondaryTensor:maxQty                 name:nil];

    // AND all conditions into a single mask
    MPSGraphTensor* c12   = [graph logicalANDWithPrimaryTensor:c1  secondaryTensor:c2  name:nil];
    MPSGraphTensor* c45   = [graph logicalANDWithPrimaryTensor:c4  secondaryTensor:c5  name:nil];
    MPSGraphTensor* c345  = [graph logicalANDWithPrimaryTensor:c3  secondaryTensor:c45 name:nil];
    MPSGraphTensor* mask  = [graph logicalANDWithPrimaryTensor:c12 secondaryTensor:c345 name:@"mask"];

    // Cast bool mask → float so it can be used in multiplication
    MPSGraphTensor* maskF   = [graph castTensor:mask toType:MPSDataTypeFloat32 name:nil];

    // revenue = SUM(price * discount * mask)
    MPSGraphTensor* product = [graph multiplicationWithPrimaryTensor:tPrice
                                                    secondaryTensor:tDiscount name:nil];
    MPSGraphTensor* masked  = [graph multiplicationWithPrimaryTensor:product
                                                    secondaryTensor:maskF     name:nil];
    MPSGraphTensor* revenue = [graph reductionSumWithTensor:masked axis:0 name:@"revenue"];

    // ----------------------------------------------------------------
    // Step 3: Wrap CPU vectors → GPU tensors (zero-copy)
    // ----------------------------------------------------------------
    MPSGraphTensorData* sdTD = columnTensor(device, (void*)shipdate.data(), N, MPSDataTypeInt32);
    MPSGraphTensorData* diTD = columnTensor(device, (void*)discount.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* qtTD = columnTensor(device, (void*)quantity.data(), N, MPSDataTypeFloat32);
    MPSGraphTensorData* prTD = columnTensor(device, (void*)price.data(),    N, MPSDataTypeFloat32);

    NSDictionary* feeds = @{
        tShipdate: sdTD,
        tDiscount: diTD,
        tQuantity: qtTD,
        tPrice:    prTD
    };

    // Output: single float (the revenue sum)
    id<MTLBuffer> outBuf = allocSharedBuffer(device, sizeof(float));
    MPSGraphTensorData* outTD = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:outBuf shape:@[@1] dataType:MPSDataTypeFloat32];
    NSMutableDictionary* results = [NSMutableDictionary dictionaryWithObject:outTD
                                                                       forKey:revenue];

    // ----------------------------------------------------------------
    // Step 4: 2 warmup runs + 1 measured run
    // ----------------------------------------------------------------
    double gpuMs = runQueryWarmedUp(graph, feeds, nil, queue, results);

    // ----------------------------------------------------------------
    // Step 5: Read result and print
    // ----------------------------------------------------------------
    float rev = *(float*)[outBuf contents];

    printf("  Revenue: $%.2f\n", rev);
    printf("  Expected (SF-1): $123,141,078.23\n");
    printTimingSummary(parseMs, gpuMs, 0.0);

    // ----------------------------------------------------------------
    // Step 6: Log to CSV
    // ----------------------------------------------------------------
    double memGB = (double)device.recommendedMaxWorkingSetSize / (1024.0 * 1024.0 * 1024.0);
    std::string sf = (g_dataset_path.find("SF-10") != std::string::npos) ? "SF-10" : "SF-1";
    logResultToCSV("results/mps_results.csv", sf, "Q6", gpuMs, 0.0,
                   std::string(device.name.UTF8String), memGB);
}
