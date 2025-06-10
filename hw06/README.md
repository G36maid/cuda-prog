# Introduction to CUDA Parallel Programming Homework Assignment 6
April, 2025

1. Histogram of a data set with exponential distribution
Write a pseudo-random number generator to generate random
numbers in (0,$\infty$ ) with the exponential distribution exp(-x), and use it
to generate a data set of 81920000 entries.

Compare the histograms computed by CPU, GPU with global memory, and GPU with shared memory, as well as their speeds.

Plot the histogram, together with the curve of the theoretical probability distribution.
Also, to determine the optimal block sizes for this problem.

## Submission Guidelines
As usual, your homework report should include your source codes,
results, and discussions (without *.exe files). The discussion file should
be prepared with a typesetting system, e.g., LaTeX, Word, etc., and it
is converted to a PDF file. All files should be zipped into one gzipped
tar file, with a file name containing your student number and the
problem set number (e.g., r05202043_ps6.tar.gz). Please send your
homework from your NTU/NTNU/NTUST email account to
twchiu@phys.ntu.edu.tw before 17:00, June 11, 2025 (deadline for all
problem sets).
If the mail server does not allow you to attach the gzipped tar file, you
can put it in the home directory of your account in twcp1, e.g.,
twcp1:/home/cuda2025/r05202043/HW6/r05202043_ps6.tar.gz
and also send email notification to me.
Animate the Moreover, ani

---

# Results and Discussion

## Performance Comparison

The histogram computation was performed using three methods: CPU, GPU with global memory, and GPU with shared memory. The timing results for various CUDA block sizes are summarized below:

| Block Size | GPU Global (ms) | GPU Shared (ms) |
|------------|-----------------|-----------------|
| 8          | 44.509          | 35.799          |
| 16         | 33.818          | 19.047          |
| 32         | 33.713          | 19.126          |
| 64         | 33.881          | 19.170          |
| 128        | 33.845          | 9.549           |
| 256        | 33.782          | 5.022           |
| 512        | 33.605          | 3.883           |
| 1024       | 33.561          | 3.986           |

- **CPU Histogram Time:** 162.851 ms
- **Optimal GPU Global Block Size:** 1024 (33.561 ms)
- **Optimal GPU Shared Block Size:** 512 (3.883 ms)
- **CPU vs GPU (global) mismatched bins:** 0
- **CPU vs GPU (shared) mismatched bins:** 0

### Analysis

- The GPU implementations are significantly faster than the CPU, especially when using shared memory. The best shared memory performance (3.883 ms) is about 42 times faster than the CPU.
- Shared memory optimization provides a substantial speedup over global memory, particularly at higher block sizes.
- The optimal block size for global memory is 1024, while for shared memory it is 512. Increasing block size generally improves performance up to a point, after which gains plateau due to hardware limits.
- Both GPU implementations produce results identical to the CPU (0 mismatched bins), confirming correctness.

### Recommendations

- For large-scale histogram computations on exponential data, using shared memory with a block size of 512 is optimal on this hardware.
- Always verify correctness when optimizing for performance.
- Visualization of the histogram and the theoretical exponential distribution should be included in the final report for completeness.

