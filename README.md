CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Jacqueline Guan
  * [LinkedIn](https://www.linkedin.com/in/jackie-guan/)
  * [Personal website](https://jyguan18.github.io/)
* Tested on my personal laptop:
  * Windows 11 Pro 26100.4946
  * Processor	AMD Ryzen 9 7945HX with Radeon Graphics
  * 32 GB RAM
  * Nvidia GeForce RTX 4080

Introduction
=============
In this project, I implemented a GPU stream compaction in CUDA, focusing on the core algorithms of parallel prefix sum (scan) and stream compaction. The primary goal was to remove zeros from an array of integers, which is a fundamental step for more complex GPU algorithms like accelerating a path tracer by removing terminated paths. The purpose of this project was to help think about data parallelism and the unique challenges and opportunities of programming for GPUs

Implementation Details
=============
### Part 1: CPU Scan and Stream Compaction
I first implemented a CPU version of the scan and stream compaction algorithms in stream_compaction/cpu.cu. The CPU scan computes an exclusive prefix sum using a simple for loop. The stream compaction methods include:
* compactWithoutScan: A direct implementation that iterates through the array and copies non-zero elements
* compactWithScan: An implementation that first maps the input to an array of 0s and 1s, then uses a scan to determine the output positions, and finally uses a scatter operation to produce the compacted array.

### Part 2: Naive GPU Scan Algorithm
I implemented the naive GPU scan in stream_compaction/naive.cu. I used global memory and multiple kernel invocations to perform the scan. A challenge was managing race conditions using two separate device arrays and swapping them at each iteration to ensure correct reads and writes. The implementation also required careful handling of ilog2ceil(n) separate kernel launches.

### Part 3: Work Efficient GPU Scan and Stream Compaction
I implemented the work-efficient algorithm in stream_compaction/efficient.cu.
* Scan: The scan algorithm operates on a binary tree structure and is implemented in-place, which avoids the race conditions of the naive approach. I ensured the implementation correctly handles non-power-of-two sized arrays by padding.
* Stream Compaction: The stream compaction uses the scan function and a scatter algorithm. I implemented the helper kernels kernMapToBoolean and kernScatter in stream_compaction/common.cu.

### Part 4: Using Thrust's Implementation
For this part, I used the Thurst library to implement both scan and stream compaction in stream_compaction/thrust.cu. The scan implementation is a simple wrapper around thrust::exclusive_scan, which demonstrates how to leverage highly optimized, existing GPU libraries.


Performance Analysis
=============
This section presents the results of my performance analysis, comparing the different scan implementations. All measurements were taken with Release mode builds and without debugging.

### Performance Plots
CPU Scan vs. GPU Scan:
![](img/general.png)

This graph compares the run times of the serial CPU scan against the Naive, Work-Efficient, and Thrust GPU scan implementations across a range of array sizes.

In this graph, we can see that all of the algorithms have different performance characteristics as the input size grows.
* Thrust is the fastest implementation, especially as the array size increases. Since it is a NVIDIA library, we can see how much faster and the high level of optimization of it.
* The Work-Efficient scan is the second fastest GPU algorithm, but it is still slower than Thrust. Even so, there's a pretty clear performance advantage at arrays larger than 2^22. Since I did not implement shared memory, every intermediate calculation must pass through the slower global memory, which makes the algorithm more memory-bandwidth bound.
* The CPU and Naive GPU scans have really similar results. In fact, for array sizes lower than 2^22, it even sometimes looks faster than thrust or Work-Efficient scan.

### Performance Bottlenecks
* Naive GPU Scan: For the Naive GPU scan, I think the primary bottleneck is using log(n) separate kernel launches.
* Work-Efficient GPU Scan: While it is faster, this algorithm still has limitations based on memory bandwidth. The up-sweep and down-sweep phases require multiple reads and writes from global memory.
* CPU Scan: This can only process one element at a time, making performance directly proportional to the array size. It can't take advantage of data parallelism so in large datasets, its performance quickly falls behind.

Extra Credit
============
### Work-Efficient Scan Optimization
A common issue when implementing a work-efficient scan is that it can perform worse than a serial CPU implementation. This happens if the GPU is not used efficiently and there are a lot of threads that are not being used.

My impelmentation did not suffer from this issue because the number of thread blocks launched for each step was dynamically scaled according to the amount of work required at that level. 
```dim3 blocksPerGrid((nPadded / 1 <<< (d + 1)) + blockSize - 1) / blockSize); ```
As d increases, the number of blocks launched decreases, which keeps the active threads busy and avoids launching a large number of "lazy" threads.

Application Output
============

```****************
** SCAN TESTS **
****************
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 16777214   0 ]
==== cpu scan, power-of-two ====
   elapsed time: 12.2615ms    (std::chrono Measured)
    [   0   0   1   3   6  10  15  21  28  36  45  55  66 ... -41943037 -25165823 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 12.1849ms    (std::chrono Measured)
    [   0   0   1   3   6  10  15  21  28  36  45  55  66 ... -92274673 -75497462 ]
    passed
==== naive scan, power-of-two ====
   elapsed time: 12.7741ms    (CUDA Measured)
    [   0   0   1   3   6  10  15  21  28  36  45  55  66 ... -41943037 -25165823 ]
    passed
==== naive scan, non-power-of-two ====
   elapsed time: 11.8661ms    (CUDA Measured)
    passed
==== work-efficient scan, power-of-two ====
   elapsed time: 5.25357ms    (CUDA Measured)
    passed
==== work-efficient scan, non-power-of-two ====
   elapsed time: 4.76275ms    (CUDA Measured)
    passed
==== thrust scan, power-of-two ====
   elapsed time: 1.71027ms    (CUDA Measured)
    passed
==== thrust scan, non-power-of-two ====
   elapsed time: 1.31456ms    (CUDA Measured)
    passed

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   0   1   2   3   4   5   6   7   8   9  10  11  12 ... 16777214   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 11.6285ms    (std::chrono Measured)
    [   1   2   3   4   5   6   7   8   9  10  11  12  13 ... 16777213 16777214 ]
    passed
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 11.2948ms    (std::chrono Measured)
    [   1   2   3   4   5   6   7   8   9  10  11  12  13 ... 16777211 16777212 ]
    passed
==== cpu compact with scan ====
   elapsed time: 58.9926ms    (std::chrono Measured)
    [   1   2   3   4   5   6   7   8   9  10  11  12  13 ... 16777213 16777214 ]
    passed
==== work-efficient compact, power-of-two ====
   elapsed time: 12.5078ms    (CUDA Measured)
    [   1   2   3   4   5   6   7   8   9  10  11  12  13 ... 16777213 16777214 ]
    passed
==== work-efficient compact, non-power-of-two ====
   elapsed time: 10.498ms    (CUDA Measured)
    [   1   2   3   4   5   6   7   8   9  10  11  12  13 ... 16777211 16777212 ]
    passed
```
