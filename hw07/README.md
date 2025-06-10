# Introduction to CUDA Parallel Programming

## Homework Assignment 7

**May 2025**

### 1. Monte Carlo integration in 10-dimensions

Write a C/C++ program to perform the Monte Carlo integration of the 10-dimensional integral:

$$
I = \int_0^1 dx_1 \int_0^1 dx_2 \cdots \int_0^1 dx_{10} \, \frac{1}{1 + x_1^2 + x_2^2 + \cdots + x_{10}^2}
$$

with the following algorithms:

#### (a) Simple sampling

#### (b) Importance sampling with the Metropolis algorithm

Using the weight function:

$$
W(x_1, x_2, \ldots, x_{10}) = w(x_1) w(x_2) \cdots w(x_{10}), \quad w(x) = Ce^{-a x}, \quad i = 1, \ldots, 10
$$

Where the normalization constant $C$ and the parameter $a$ are determined by yourself.

In each case, compute the **mean** and **standard deviation** versus the number of samplings:

$$
N = 2^n, \quad n = 2, 3, \ldots, 16
$$

Write CUDA code to perform this Monte Carlo integration with multi-GPUs.
Test your code with
one and two GPUs,
by comparing GPU and CPU results.
You may start by
writing your CUDA code for one GPU

### Submission Guidelines

Your homework report should include:

* Source code (C/C++ and CUDA)
* Results (including performance comparison)
* Discussion (prepared using a typesetting system such as LaTeX, Word, etc., and converted to PDF)

All files should be compressed into a single gzipped tar file.
**Naming format:** `r05202043_HW7.tar.gz` (replace with your own student ID).
Submit the file via your **NTU/NTNU/NTUST email account** to:

```
twchiu@phys.ntu.edu.tw
```

**Deadline:** Before 17:00, June 11, 2025

If email submission fails (due to file size), upload it to your home directory on `twcp1`, e.g.:

```
twcp1:/home/cuda2025/R05202043/HW7/R05202043_HW7.tar.gz
```

And **send an email notification** to the instructor as well.
