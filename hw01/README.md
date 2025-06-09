# Introduction to CUDA Parallel Programming Homework Assignment 1
March, 2025

## Problem Statement
1. To write your own CPU+GPU code for the modified matrix addition as
defined by c(i,j) = 1/a(i,j) + 1/b(i,j). Setting the input NxN matrices A
and B (where N=6400) with entries of random numbers between 0.0
and 1.0, determine the optimal block size by running your code. You
can use the sample code

twqcp1:/home/cuda_lecture_2025/vecAdd_1GPU/vecAdd.cu as a
template to develop your own code.

## Results and Analysis

### Performance Analysis
We tested different thread block sizes (32, 64, 128, 256, 512, 1024) to find the optimal configuration. Here are the key findings:

#### CPU Performance Baseline:
- Processing time: 99.51 ms
- GFLOPS: 1.65

#### GPU Performance for Different Block Sizes:
| Block Size | Kernel Time (ms) | GFLOPS | Total Time (ms) |
|------------|-----------------|---------|----------------|
| 32         | 9.76           | 16.79   | 179.54        |
| 64         | 6.63           | 24.72   | 139.31        |
| 128        | 6.53           | 25.08   | 139.13        |
| 256        | 6.49           | 25.24   | 139.94        |
| 512        | 6.52           | 25.14   | 139.22        |
| 1024       | 6.59           | 24.85   | 138.96        |

#### Optimal Configuration:
- Best Block Size: 256 threads
- Best Kernel Time: 6.49 ms
- Peak Performance: 25.24 GFLOPS
- Speedup vs CPU: 15.33x

### Memory Transfer Analysis:
- Input Transfer Time: ~91-92 ms
- Output Transfer Time: ~41 ms
- Total GPU Time (including transfers): ~139-180 ms

### Key Observations:
1. The optimal block size of 256 threads provides the best balance between parallelism and resource utilization
2. Memory transfers dominate the total execution time
3. Small block size (32) shows significantly worse performance
4. Block sizes larger than 256 show slightly degraded performance
5. Result validation confirms numerical accuracy (norm difference = 0)

## Submission

Your homework report should include your source codes, results, and
discussions. The discussion file should be prepared with a typesetting
system, e.g., LaTeX, Word, etc., and it is converted to a PDF file. All
files should be zipped into one gzipped tar file, with a file name
containing your student number and the problem set number
(e.g., r05202043_HW1.tar.gz). Please send your homework with the
title "your_student_number_HW1" to twchiu@phys.ntu.edu.tw
before 17:00, June 11, 2025 (deadline for all problem sets).