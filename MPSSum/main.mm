/**
 * MetalVectorSum - TPC-H o_totalprice Sum
 *
 * Reads o_totalprice from TPC-H orders.tbl and computes the sum
 * using Metal Performance Shaders, matching the Python/PyTorch implementation.
 *
 * Strategy:
 *   Treat the input vector as a 1×N matrix (A).
 *   Create an N×1 matrix of ones (B).
 *   Use MPSMatrixMultiplication to compute C = A × B.
 *   C is a 1×1 matrix whose single element equals sum(A).
 */

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {

        // ------------------------------------------------------------------ //
        // 1. Metal device + command queue
        // ------------------------------------------------------------------ //
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            NSLog(@"[ERROR] Metal is not supported on this device.");
            return EXIT_FAILURE;
        }
        NSLog(@"[INFO]  Device : %@", device.name);

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

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
            NSLog(@"[ERROR] Run common/setupAndRun.sh first to generate the TPC-H data.");
            return EXIT_FAILURE;
        }

        NSArray<NSString *> *lines = [content componentsSeparatedByString:@"\n"];
        NSMutableData *priceData = [NSMutableData data];
        float cpuSum = 0.0f;

        for (NSString *line in lines) {
            if (line.length == 0) continue;
            NSArray<NSString *> *fields = [line componentsSeparatedByString:@"|"];
            if (fields.count > 3) {
                float price = [fields[3] floatValue];
                cpuSum += price;
                [priceData appendBytes:&price length:sizeof(float)];
            }
        }

        NSUInteger N = priceData.length / sizeof(float);
        NSUInteger floatSize = sizeof(float);

        NSLog(@"[INFO]  Rows   : %lu", (unsigned long)N);
        NSLog(@"[INFO]  CPU sum: %.2f  (expected)", cpuSum);

        // ------------------------------------------------------------------ //
        // 3. Metal buffers
        // ------------------------------------------------------------------ //
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
        // 4. MPS matrix descriptors
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

        // ------------------------------------------------------------------ //
        // 5. MPS matrix wrappers
        // ------------------------------------------------------------------ //
        MPSMatrix *matA = [[MPSMatrix alloc] initWithBuffer:inputBuf  descriptor:descA];
        MPSMatrix *matB = [[MPSMatrix alloc] initWithBuffer:onesBuf   descriptor:descB];
        MPSMatrix *matC = [[MPSMatrix alloc] initWithBuffer:outputBuf descriptor:descC];

        // ------------------------------------------------------------------ //
        // 6. MPSMatrixMultiplication kernel  C = 1.0 * A × B + 0.0 * C
        // ------------------------------------------------------------------ //
        MPSMatrixMultiplication *matMul =
            [[MPSMatrixMultiplication alloc] initWithDevice:device
                                              transposeLeft:NO
                                             transposeRight:NO
                                                resultRows:1
                                             resultColumns:1
                                           interiorColumns:N
                                                     alpha:1.0
                                                      beta:0.0];

        // ------------------------------------------------------------------ //
        // 7. Encode and commit to the GPU
        // ------------------------------------------------------------------ //
        id<MTLCommandBuffer> cmdBuf = [commandQueue commandBuffer];
        [matMul encodeToCommandBuffer:cmdBuf
                           leftMatrix:matA
                          rightMatrix:matB
                        resultMatrix:matC];
        [cmdBuf commit];
        [cmdBuf waitUntilCompleted];

        // ------------------------------------------------------------------ //
        // 8. Read result and verify (0.1% relative tolerance for float32)
        // ------------------------------------------------------------------ //
        float gpuSum = *((float *)outputBuf.contents);
        float relErr = fabsf(gpuSum - cpuSum) / fabsf(cpuSum);

        NSLog(@"[INFO]  GPU sum: %.2f  (MPS computed)", gpuSum);

        if (relErr < 1e-3f) {
            NSLog(@"[PASS]  GPU sum matches CPU sum (rel err: %.6f)", relErr);
        } else {
            NSLog(@"[FAIL]  Mismatch! GPU=%.2f  CPU=%.2f  (rel err: %.6f)", gpuSum, cpuSum, relErr);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
