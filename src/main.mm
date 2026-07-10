#import "mps_infra.h"
#import <Foundation/Foundation.h>

void runMicrobenchmarks(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ1(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ3(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ4(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ5(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ6(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ9(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ12(id<MTLDevice> device, id<MTLCommandQueue> queue);
void runQ14(id<MTLDevice> device, id<MTLCommandQueue> queue);

static void showHelp() {
    printf("GPU Database MPS Benchmark\n");
    printf("Usage: GPUDBMPSBenchmark [sf1|sf10] [query]\n\n");
    printf("Available queries:\n");
    printf("  all           - Run all benchmarks (default)\n");
    printf("  micro         - Micro-benchmarks (selection, aggregation, join)\n");
    printf("  q1            - TPC-H Q1  (Pricing Summary)\n");
    printf("  q3            - TPC-H Q3  (Shipping Priority)      [3-way join]\n");
    printf("  q4            - TPC-H Q4  (Order Priority Checking) [semi-join]\n");
    printf("  q5            - TPC-H Q5  (Local Supplier Volume)  [5-way join]\n");
    printf("  q6            - TPC-H Q6  (Forecasting Revenue Change)\n");
    printf("  q9            - TPC-H Q9  (Product Type Profit)    [6-table join]\n");
    printf("  q12           - TPC-H Q12 (Shipping Modes)\n");
    printf("  q14           - TPC-H Q14 (Promotion Effect)\n");
    printf("  help          - Show this help message\n\n");
    printf("Scale Factors:\n");
    printf("  sf1           - TPC-H SF-1  (~6M lineitem rows)\n");
    printf("  sf10          - TPC-H SF-10 (~60M lineitem rows)\n");
    printf("Examples:\n");
    printf("  GPUDBMPSBenchmark              # All benchmarks on SF-1\n");
    printf("  GPUDBMPSBenchmark q6           # Q6 on SF-1\n");
}

int main(int argc, const char* argv[]) {
    @autoreleasepool {
        std::string query = "all";

        for (int i = 1; i < argc; ++i) {
            std::string arg(argv[i]);
            if (arg == "help" || arg == "--help" || arg == "-h") {
                showHelp();
                return 0;
            }
            if (arg == "sf1")   { g_dataset_path = "data/SF-1/";   g_sf100_mode = false; continue; }
            if (arg == "sf10")  { g_dataset_path = "data/SF-10/";  g_sf100_mode = false; continue; }
            query = arg;
        }

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Error: No Metal device found.\n");
            return 1;
        }

        id<MTLCommandQueue> commandQueue __attribute__((unused)) = [device newCommandQueue];

        printf("=== GPU Database MPS Benchmark ===\n");
        printf("GPU:        %s\n", device.name.UTF8String);
        printf("Scale:      %s\n", g_dataset_path.c_str());
        printf("Mode:       %s\n", g_sf100_mode ? "SF-100 chunked streaming" : "standard");
        printf("Query:      %s\n", query.c_str());
        printf("----------------------------------\n");

        if (query == "micro" || query == "all")
            runMicrobenchmarks(device, commandQueue);
        if (query == "q1" || query == "all")
            runQ1(device, commandQueue);
        if (query == "q3" || query == "all")
            runQ3(device, commandQueue);
        if (query == "q4" || query == "all")
            runQ4(device, commandQueue);
        if (query == "q5" || query == "all")
            runQ5(device, commandQueue);
        if (query == "q6" || query == "all")
            runQ6(device, commandQueue);
        if (query == "q9" || query == "all")
            runQ9(device, commandQueue);
        if (query == "q12" || query == "all")
            runQ12(device, commandQueue);
        if (query == "q14" || query == "all")
            runQ14(device, commandQueue);
    }
    return 0;
}
