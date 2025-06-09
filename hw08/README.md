Introduction to CUDA Parallel Programming Homework Assignment 8
May 2025
1. GPU accelerated Monte Carlo simulation of 2D Ising model on a torus
For one GPU, determine the optimal block size for MC simulation on the
200 × 200 lattice. Then use the optimal block size to perform MC simulations at
B=0, and T=2.0, 2.1, 2.2, 2.3, 2.4, and 2.5, and measure < E > and < M >, and
estimate their errors respectively. Summarize your results of < E >, δ < E >,
< M >, and δ < M > with tables, and plot them versus T.
2. CUDA C/C++ code for multi-GPUs
Write a CUDA C/C++ code for simulation of 2D Ising model on the torus with
multi-GPUs. Test your code with one and two GPUs, by comparing GPU and CPU
results.
3. Repeat 1. for the MC simulation with 2 GPUs

## Submission Guidelines

As usual, your homework report should include your source codes, results, and
discussions. The discussion file should be prepared with a typesetting system, e.g.,
LaTeX, Word, etc., and it is converted to a PDF file. All files should be zipped into one
gzipped tar file, with a file name containing your student number and the problem
set number (e.g., r05202043_HW8.tar.gz). Please send your homework from your
NTU/NTNU/NTUST email account to twchiu@phys.ntu.edu.tw before 17:00, June 11,
2025 (deadline for all problem sets).
If the mail server does not allow you to attach the gzipped tar file, you can put it in
the home directory of your account in twcp1, e.g.,
twcp1:/home/cuda2024/R05202043/HW8/R05202043_HW8.tar.gz
and also send email notification to me
