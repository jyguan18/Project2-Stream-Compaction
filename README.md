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

This graph compares the run times of the serial CPU scan against the Naive, Work-Efficient, and Thrust GPU scan implementations.

### Performance Bottlenecks

Extra Credit
============
