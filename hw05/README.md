# Introduction to CUDA Parallel Programming Homework Assignment 5
April, 2025

1. Heat Diffusion
Using a Cartesian grid of 1024 x 1024, solve for the thermal
equilibrium temperature distribution on a square plate. The
temperature along the top edge of the plate is at 400 K, while the
remainder of the circumference is at 273 K. Write a CUDA code for
multi-GPUs to solve this problem.

Test your code with one and two
GPUs. Also, to determine the optimal block size for this problem. The
value of $\omega$ can be fixed to 1.

## Heat Diffusion Equation

### Mathematical Formulation

For steady-state two-dimensional heat conduction, the temperature distribution T(x,y) satisfies the two-dimensional Laplace equation:

$$\frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

Or equivalently:

$$\nabla^2 T = 0$$

### Boundary Conditions

Based on the problem description:
- Top boundary: T(x, L) = 400 K
- Other three boundaries: T(0, y) = T(L, y) = T(x, 0) = 273 K

where L is the side length of the square plate.

### Numerical Solution Method

**Finite Difference Discretization**

Using the five-point finite difference stencil for grid point (i,j):

$$T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1} - 4T_{i,j} = 0$$

**Iterative Solution with SOR**

Successive Over-Relaxation (SOR) method for iterative solving:

$$T_{i,j}^{(k+1)} = (1-\omega)T_{i,j}^{(k)} + \frac{\omega}{4}(T_{i+1,j}^{(k)} + T_{i-1,j}^{(k+1)} + T_{i,j+1}^{(k)} + T_{i,j-1}^{(k+1)})$$

Where:
- T_{i,j}^{(k)} represents temperature at grid point (i,j) during k-th iteration
- ω is the relaxation factor, fixed to 1 (Gauss-Seidel method)
- Iteration continues until convergence criteria is met

**Convergence Criteria**

Convergence is achieved when the maximum temperature difference between consecutive iterations is below a threshold:

$$\max_{i,j} |T_{i,j}^{(k+1)} - T_{i,j}^{(k)}| < \epsilon$$

where ε is the predefined convergence tolerance (e.g., 10^{-6}).

## Implementation Considerations

### CUDA Optimization Strategies

- **Memory Access Patterns**: Optimize coalesced memory access for better performance
- **Block Size Tuning**: Experiment with different block dimensions to find optimal configuration
- **Shared Memory Usage**: Utilize shared memory for frequently accessed neighboring values


## Remarks on using texture for multiGPUs
### Texture Objects and Peer Access in CUDA

In CUDA, texture objects are bound to memory that resides on the same
local device. The CUDA runtime does not support creating a texture object
on one GPU that references global memory physically located on a peer GPU,
even when peer-to-peer (P2P) access is enabled between devices.

Specifically, texture memory binding is strictly local:
a GPU can only bind texture objects to memory it directly owns.
Therefore, when implementing multi-GPU applications—such
as solving the Laplace equation with fixed boundary conditions—it is
generally not recommended to use texture memory for data residing on
remote (peer) GPUs.

For such multi-GPU applications, it is preferable to rely on global memory
access with cudaMemcpyPeer or unified memory, combined with explicit
memory management and synchronization.

## Submission Guidelines
As usual, your homework report should include your source codes,
results, and discussions (without any executable files). The discussion
file should be prepared with a typesetting system, e.g., LaTeX, Word,
etc., and it is converted to a PDF file. All files should be zipped into one
gzipped tar file, with a file name containing your student number and
the problem set number (e.g., r05202043_ps5.tar.gz). Please send
your homework with the title “your_student_number_HW5” to
twchiu@phys.ntu.edu.tw before 17:00, June 11, 2025 (deadline for all
problem sets).
