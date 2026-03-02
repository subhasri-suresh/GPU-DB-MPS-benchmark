/**
 * MetalVectorSum
 *
 * Computes the sum of all elements in a vector using Metal Performance Shaders.
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
        // 2. Sample vector  (1 million elements, all 1.0)
        // ------------------------------------------------------------------ //
        NSUInteger N = 1000000;
        float *inputData = (float *)malloc(N * sizeof(float));
        for (NSUInteger i = 0; i < N; i++) inputData[i] = 1.0f;

        float cpuSum = 0.0f;
        for (NSUInteger i = 0; i < N; i++) cpuSum += inputData[i];

        NSLog(@"[INFO]  N      : %lu", (unsigned long)N);
        NSLog(@"[INFO]  CPU sum: %.1f  (expected)", cpuSum);

        // ------------------------------------------------------------------ //
        // 3. Allocate Metal buffers
        //      inputBuf  — 1×N matrix (the vector as a row)
        //      onesBuf   — N×1 matrix (all 1.0)
        //      outputBuf — 1×1 matrix (the result)
        // ------------------------------------------------------------------ //
        NSUInteger floatSize = sizeof(float);

        id<MTLBuffer> inputBuf =
            [device newBufferWithBytes:inputData
                                length:N * floatSize
                               options:MTLResourceStorageModeShared];
        free(inputData);

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
        // 8. Read result and verify
        // ------------------------------------------------------------------ //
        float gpuSum = *((float *)outputBuf.contents);

        NSLog(@"[INFO]  GPU sum: %.1f  (MPS computed)", gpuSum);

        if (fabsf(gpuSum - cpuSum) < 1e-3f) {
            NSLog(@"[PASS]  GPU sum matches CPU sum.");
        } else {
            NSLog(@"[FAIL]  Mismatch! GPU=%.4f  CPU=%.4f", gpuSum, cpuSum);
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
