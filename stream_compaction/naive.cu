#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void naiveScan(int n, int *odata, const int *idata, int d) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx >= n) return;
            if (idx >= (1 << (d - 1))) {
                odata[idx] = idata[idx - (1 << (d - 1))] + idata[idx];
            }
            else {
                odata[idx] = idata[idx];
            }
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int* tempIn;
            int* tempOut;
            cudaMalloc((void**)&tempIn, n * sizeof(int));
            cudaMalloc((void**)&tempOut, n * sizeof(int));
            cudaMemset(tempIn, 0, sizeof(int));
            cudaMemcpy(tempIn + 1, idata, (n - 1) * sizeof(int), cudaMemcpyHostToDevice);
            
            for (int d = 1; d <= ilog2ceil(n); ++d) {
                naiveScan << < 1, std::min(1024, n) >> > (n, tempOut, tempIn, d); // Check that this is the right blocksize!!
                std::swap(tempOut, tempIn);
            }

            std::swap(tempOut, tempIn);

            cudaMemcpy(odata, tempOut, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(tempIn);
            cudaFree(tempOut);

            timer().endGpuTimer();
        }
    }
}
