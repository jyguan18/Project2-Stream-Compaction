#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"
#include <iostream>

#define blockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */

        __global__ void kernelUpSweep(int n, int* odata, int d) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            int k = (1 << (d + 1)) * idx;

            if (k >= n) return;
            odata[k + (1 << (d + 1)) - 1] += odata[k + (1 << d) - 1];
        }

        __global__ void kernelDownSweep(int n, int* odata, int d) {
            int idx = threadIdx.x + (blockIdx.x * blockDim.x);
            int k = (1 << (d + 1)) * idx;

            if (k >= n) return;
            int t = odata[k + (1 << d) - 1];
            odata[k + (1 << d) - 1] = odata[k + (1 << (d + 1)) - 1];
            odata[k + (1 << (d + 1)) - 1] += t;

        }

        void scan(int n, int *odata, const int *idata, bool isCompact) {
            if (!isCompact) {
                timer().startGpuTimer();
            }

            int logn = ilog2ceil(n);
            int nPadded = 1 << logn;

            int *dev_data;
            cudaMalloc((void**)&dev_data, nPadded * sizeof(int));

            cudaMemset(dev_data, 0, nPadded * sizeof(int));

            if (isCompact) {
                cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            else {
                cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            }
            
            
            for (int d = 0; d < logn; ++d) {
                // n / (2 ^ (d + 1))
                dim3 blocksPerGrid((nPadded / (1 << (d + 1)) + blockSize - 1) / blockSize);

                kernelUpSweep << < blocksPerGrid, blockSize >> > (nPadded, dev_data, d);
            }

            cudaMemset(dev_data + (nPadded - 1), 0, sizeof(int));

            for (int d = logn - 1; d >= 0; --d) {
                // n / (2 ^ (d + 1))
                dim3 blocksPerGrid((nPadded / (1 << (d + 1)) + blockSize - 1) / blockSize);

                kernelDownSweep << < blocksPerGrid, blockSize >> > (nPadded, dev_data, d);

            }

            if (isCompact) {
                cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToDevice);
            }
            else {
                cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);
            }

            cudaFree(dev_data);

            if (!isCompact) {
                timer().endGpuTimer();
            }
        }

        void printArray(int n, int* a, bool abridged = false) {
            printf("    [ ");
            for (int i = 0; i < n; i++) {
                if (abridged && i + 2 == 15 && n > 16) {
                    i = n - 2;
                    printf("... ");
                }
                printf("%3d ", a[i]);
            }
            printf("]\n");
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            const size_t bytes = n * sizeof(int);
            cudaError_t cpyRes;

            // mark em
            int* dev_Bools;
            cudaMalloc((void**)&dev_Bools, bytes);

            int* dev_idata;
            cudaMalloc((void**)&dev_idata, bytes);
            cpyRes = cudaMemcpy(dev_idata, idata, bytes, cudaMemcpyHostToDevice);
            if (cpyRes != CUDA_SUCCESS) {
                std::cout << "Copy idata failed." << std::endl;
                return -1;
            }

            int* dev_odata;
            cudaMalloc((void**)&dev_odata, bytes);

            int gridSize = (n + blockSize - 1) / blockSize;
            Common::kernMapToBoolean << < gridSize, blockSize >> > (n, dev_Bools, dev_idata);

            // scan em
            int* scanData;
            cudaMalloc((void**)&scanData, bytes);

            scan(n, scanData, dev_Bools, true);
            
            // scatter em
            Common::kernScatter << < gridSize, blockSize >> > (n, dev_odata, dev_idata, dev_Bools, scanData);
            
            cudaMemcpy(odata, dev_odata, bytes, cudaMemcpyDeviceToHost);

            int lastBool, lastScan;

            cudaMemcpy(&lastBool, dev_Bools + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastScan, scanData + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            timer().endGpuTimer();

            cudaFree(dev_Bools);
            cudaFree(scanData);

            return lastBool + lastScan;
        }
    }
}
