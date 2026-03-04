/**
 * MPSGraph o_totalprice Sum
 *
 * Reads o_totalprice from TPC-H orders.tbl and computes the sum
 * using MPSGraph — a higher-level graph API similar to PyTorch tensors.
 *
 * Strategy:
 *   Define a graph: placeholder[N] → reductionSum(axis=0) → scalar
 *   Feed the price data, run the graph, read the scalar result.
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <time.h>

static double msNow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {

        // ------------------------------------------------------------------ //
        // 1. Metal device
        // ------------------------------------------------------------------ //
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"[ERROR] Metal is not supported on this device.");
            return EXIT_FAILURE;
        }
        NSLog(@"[INFO]  Device : %@", device.name);

        // ------------------------------------------------------------------ //
        // 2. Load o_totalprice from TPC-H orders.tbl
        //    Format: o_orderkey|o_custkey|o_orderstatus|o_totalprice|...
        //    o_totalprice is field index 3 (pipe-separated)
        // ------------------------------------------------------------------ //
        NSString *filePath = @"../common/tpch-dbgen/orders.tbl";
        NSError *error = nil;
        NSString *content = [NSString stringWithContentsOfFile:filePath
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (!content) {
            NSLog(@"[ERROR] Could not read %@: %@", filePath, error.localizedDescription);
            NSLog(@"[HINT]  Run python/setupAndRun.sh first to generate TPC-H data.");
            return EXIT_FAILURE;
        }

        NSArray<NSString *> *lines = [content componentsSeparatedByString:@"\n"];
        NSMutableData *priceData = [NSMutableData data];
        float cpuSum = 0.0f;

        double cpuStart = msNow();
        for (NSString *line in lines) {
            if (line.length == 0) continue;
            NSArray<NSString *> *fields = [line componentsSeparatedByString:@"|"];
            if (fields.count > 3) {
                float price = [fields[3] floatValue];
                cpuSum += price;
                [priceData appendBytes:&price length:sizeof(float)];
            }
        }
        double cpuMs = msNow() - cpuStart;

        NSUInteger N = priceData.length / sizeof(float);
        NSLog(@"[INFO]  Rows   : %lu", (unsigned long)N);
        NSLog(@"[INFO]  CPU sum: %.2f  (expected)", cpuSum);

        // ------------------------------------------------------------------ //
        // 3. Build MPSGraph: placeholder[N] → reductionSum(axis=0) → scalar
        // ------------------------------------------------------------------ //
        MPSGraph *graph = [[MPSGraph alloc] init];

        MPSGraphTensor *inputTensor =
            [graph placeholderWithShape:@[@(N)]
                               dataType:MPSDataTypeFloat32
                                   name:@"prices"];

        MPSGraphTensor *sumTensor =
            [graph reductionSumWithTensor:inputTensor
                                     axes:@[@0]
                                     name:@"total_sum"];

        // ------------------------------------------------------------------ //
        // 4. Wrap input data as MPSGraphTensorData via MTLBuffer
        // ------------------------------------------------------------------ //
        id<MTLBuffer> inputBuf =
            [device newBufferWithBytes:priceData.bytes
                                length:priceData.length
                               options:MTLResourceStorageModeShared];

        MPSGraphTensorData *inputTensorData =
            [[MPSGraphTensorData alloc] initWithMTLBuffer:inputBuf
                                                    shape:@[@(N)]
                                                 dataType:MPSDataTypeFloat32];

        // ------------------------------------------------------------------ //
        // 5. Run the graph
        // ------------------------------------------------------------------ //
        double gpuStart = msNow();
        NSDictionary<MPSGraphTensor *, MPSGraphTensorData *> *results =
            [graph runWithFeeds:@{inputTensor: inputTensorData}
                  targetTensors:@[sumTensor]
               targetOperations:nil];

        // ------------------------------------------------------------------ //
        // 6. Read scalar result and verify (0.1% relative tolerance)
        // ------------------------------------------------------------------ //
        float gpuSum = 0.0f;
        [results[sumTensor].mpsndarray readBytes:&gpuSum strideBytes:nil];
        double gpuMs = msNow() - gpuStart;

        float relErr = fabsf(gpuSum - cpuSum) / fabsf(cpuSum);
        NSLog(@"[INFO]  GPU sum: %.2f  (MPSGraph computed)", gpuSum);
        NSLog(@"[INFO]  CPU time: %.3f ms", cpuMs);
        NSLog(@"[INFO]  GPU time: %.3f ms", gpuMs);

        if (relErr < 1e-3f) {
            NSLog(@"[PASS]  GPU sum matches CPU sum (rel err: %.6f)", relErr);
        } else {
            NSLog(@"[FAIL]  Mismatch! GPU=%.2f  CPU=%.2f  (rel err: %.6f)", gpuSum, cpuSum, relErr);
            return EXIT_FAILURE;
        }

        // ------------------------------------------------------------------ //
        // 7. Write results to file
        // ------------------------------------------------------------------ //
        NSString *result = [NSString stringWithFormat:
            @"Device: %@\nRows: %lu\nCPU sum: %.2f\nGPU sum: %.2f\nRel error: %.6f\nCPU time: %.3f ms\nGPU time: %.3f ms\n",
            device.name, (unsigned long)N, cpuSum, gpuSum, relErr, cpuMs, gpuMs];

        NSError *writeError = nil;
        [result writeToFile:@"output.txt"
                 atomically:YES
                   encoding:NSUTF8StringEncoding
                      error:&writeError];

        if (writeError) {
            NSLog(@"[ERROR] Failed to write output: %@", writeError.localizedDescription);
        } else {
            NSLog(@"[INFO]  Results written to output.txt");
        }
    }
    return EXIT_SUCCESS;
}
