/**
 * MetalVectorSum - TPC-H o_totalprice Sum
 *
 * Reads o_totalprice from TPC-H orders.tbl and computes the sum
 * using Metal Performance Shaders, using matrix
 *
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <time.h>

static double msNow(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, const char *argv[]) {
    @autoreleasepool {

        // 1. Metal device + command queue
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"[ERROR] Metal is not supported on this device.");
            return EXIT_FAILURE;
        }
        NSLog(@"[INFO]  Device : %@", device.name);

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        // 2. Load o_totalprice from TPC-H orders.tbl
        NSString *filePath = @"../common/tpch-dbgen/orders.tbl";
        NSError *error = nil;
        NSString *content = [NSString stringWithContentsOfFile:filePath
                                                      encoding:NSUTF8StringEncoding
                                                         error:&error];
        if (!content) {
            NSLog(@"[ERROR] Run common/setupAndRun.sh first to generate the TPC-H data.");
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
        NSUInteger floatSize = sizeof(float);

        NSLog(@"[INFO]  Rows   : %lu", (unsigned long)N);
        NSLog(@"[INFO]  CPU sum: %.2f  (expected)", cpuSum);

        id<MTLBuffer> inputBuf =
            [device newBufferWithBytes:priceData.bytes
                                length:priceData.length
                               options:MTLResourceStorageModeShared];

        id<MTLBuffer> onesBuf =
            [device newBufferWithLength:N * floatSize
                                options:MTLResourceStorageModeShared];
        float *onesPtr = (float *)onesBuf.contents;
        for (NSUInteger i = 0; i < N; i++) onesPtr[i] = 1.0f;

        id<MTLBuffer> outputBuf =
            [device newBufferWithLength:floatSize
                                options:MTLResourceStorageModeShared];

        // ------------------------------------------------------------------ //
        // MPS matrix descriptors
        //      A : 1 × N   rowBytes = N * sizeof(float)
        //      B : N × 1   rowBytes =     sizeof(float)
        //      C : 1 × 1   rowBytes =     sizeof(float)
        // ------------------------------------------------------------------ //
        MPSMatrixDescriptor *descA =
            [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                  columns:N
                                                 rowBytes:N * floatSize
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descB =
            [MPSMatrixDescriptor matrixDescriptorWithRows:N
                                                  columns:1
                                                 rowBytes:floatSize
                                                 dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor *descC =
            [MPSMatrixDescriptor matrixDescriptorWithRows:1
                                                  columns:1
                                                 rowBytes:floatSize
                                                 dataType:MPSDataTypeFloat32];


        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:inputBuf  descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:onesBuf   descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:outputBuf descriptor:descC];

        // MPSMatrixMultiplication kernel  C = 1.0 * A × B + 0.0 * C
        MPSMatrixMultiplication *matMul =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:NO
                                             transposeRight:NO
                                                resultRows:1
                                             resultColumns:1
                                           interiorColumns:N
                                                     alpha:1.0
                                                      beta:0.0];

        //  Encode and commit to the GPU
        double gpuStart = msNow();
        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        [matMul encodeToCommandBuffer:cmdBuf
                           leftMatrix:matA
                          rightMatrix:matB
                        resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];
        double gpuMs = msNow() - gpuStart;

        float gpuSum = *((float *)outputBuf.contents);
        float relErr = fabsf(gpuSum - cpuSum) / fabsf(cpuSum);

        NSLog(@"[INFO]  GPU sum: %.2f  (MPS computed)", gpuSum);
        NSLog(@"[INFO]  CPU time: %.3f ms", cpuMs);
        NSLog(@"[INFO]  GPU time: %.3f ms", gpuMs);

        if (relErr < 1e-3f) {
            NSLog(@"[PASS]  GPU sum matches CPU sum (rel err: %.6f)", relErr);
        } else {
            NSLog(@"[FAIL]  Mismatch! GPU=%.2f  CPU=%.2f  (rel err: %.6f)", gpuSum, cpuSum, relErr);
            return EXIT_FAILURE;
        }

        // 9. Write results to file
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
