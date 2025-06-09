Introduction to CUDA Parallel Programming Homework Assignment 9
May 2025

1. Poisson equation on the 3-dimensional lattice
Consider a point charge at the origin of the 3-dimensional lattice with periodic
boundary condition in all directions. Use cuFFT to perform the inverse Fourier
transform from the momentum space to the position space, and obtain the
potential along the diagonal as well as the x-axis of the 32 x 32 x 32 lattice.
Then, to assert that your solution is indeed physically correct.
Next, to investigate what is the largest 3D lattice you can solve the Poisson
equation with one Nvidia GTX-1060.


## Submission Guidelines

As usual, your homework report should include your source codes, results, and
discussions. The discussion file should be prepared with a typesetting system, e.g.,
LaTeX, Word, etc., and it is converted to a PDF file. All files should be zipped into one
gzipped tar file, with a file name containing your student number and the problem
set number (e.g., r05202043_HW9.tar.gz). Please send your homework from your
NTU/NTNU/NTUST email account to twchiu@phys.ntu.edu.tw before 17:00, June 11,
2025 (deadline for all problem sets).
If the mail server does not allow you to attach the gzipped tar file, you can put it in
the home directory of your account in twcp1, e.g.,
twcp1:/home/cuda2025/R05202043/HW9/R05202043_HW9.tar.gz
and also send email notification to me.
