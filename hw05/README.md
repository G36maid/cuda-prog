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

### Mathematical Foundation

For steady-state heat conduction in a 2D domain, the temperature distribution satisfies the Laplace equation:

$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0$$

where $T(x,y)$ represents the temperature at position $(x,y)$.

### Numerical Discretization

Using finite difference method with central differences on a uniform grid with spacing $h$:

$$\frac{\partial^2 T}{\partial x^2} \approx \frac{T_{i+1,j} - 2T_{i,j} + T_{i-1,j}}{h^2}$$

$$\frac{\partial^2 T}{\partial y^2} \approx \frac{T_{i,j+1} - 2T_{i,j} + T_{i,j-1}}{h^2}$$

This leads to the five-point stencil formula:

$$T_{i,j} = \frac{1}{4}(T_{i+1,j} + T_{i-1,j} + T_{i,j+1} + T_{i,j-1})$$

### Iterative Solution: Jacobi Method

The Jacobi iteration scheme updates all grid points simultaneously:

$$T_{i,j}^{(k+1)} = \frac{1}{4}(T_{i+1,j}^{(k)} + T_{i-1,j}^{(k)} + T_{i,j+1}^{(k)} + T_{i,j-1}^{(k)})$$

where superscript $(k)$ denotes the iteration number.

### CUDA Implementation

**Jacobi Kernel:**
```cuda
__global__ void jacobi_kernel(
    float* T_new,
    const float* T_old,
    int width, int height
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i > 0 && i < width-1 && j > 0 && j < height-1) {
        T_new[j*width + i] = 0.25f * (
            T_old[j*width + (i+1)] +
            T_old[j*width + (i-1)] +
            T_old[(j+1)*width + i] +
            T_old[(j-1)*width + i]
        );
    }
}
```

**Boundary Conditions:**
- Top edge: $T = 400$ K
- Left, right, bottom edges: $T = 273$ K

**Convergence Criterion:**
$$\max_{i,j} |T_{i,j}^{(k+1)} - T_{i,j}^{(k)}| < \epsilon$$

where $\epsilon = 10^{-6}$ is the tolerance.

### Multi-GPU Strategy

For multi-GPU implementation:

1. **Domain Decomposition**: Split the 1024×1024 grid horizontally between GPUs
2. **Boundary Exchange**: Use `cudaMemcpyPeer` for halo region communication
3. **Synchronization**: Coordinate iterations across devices

**Performance Optimization:**
- Test block sizes: 16×16, 32×32, 64×64, and more
- Use shared memory for stencil operations
- Overlap computation with communication

maybe can use cudaMemcpyPeer to Exchange boundary data between GPUs
```
# spilt
+-------------------+
| GPU 0            |
|===================| <-- boundary
| GPU 1            |
+-------------------+
```
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
your homework with the title "your_student_number_HW5" to
twchiu@phys.ntu.edu.tw before 17:00, June 11, 2025 (deadline for all
problem sets).
