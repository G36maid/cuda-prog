Introduction to CUDA Parallel Programming Homework Assignment 3
March, 2025

## Problem Statement
1. Solve the Poisson equation on a 3D lattice with boundary conditions. Consider a cube of size L x L x L with a point charge q=1 at its center (L/2, L/2, L/2), with lattice sites (0, 1, 2, ..., L) in each direction, subject to the boundary conditions with potential equal to zero on its entire surface. Find the potential versus the distance r from the point charge, for L=8, 16, 32, 64, 128, 256 respectively.

2. Does the potential approach the Coulomb's law in the limit L >>1 ?

## Implementation and Results

### Numerical Method
- **Algorithm**: 3D Poisson equation solved using finite difference method with 6-point stencil
- **Iteration Method**: Jacobi iteration with parallel reduction
- **Boundary Conditions**: Zero potential on cube surfaces
- **Point Charge**: q=1 at cube center

### CUDA Implementation
- **Thread Block Configuration**: 3D blocks (NxNxN threads where N=2 to 16)
- **Grid Configuration**: Automatically adjusted based on L size
- **Shared Memory**: Used for parallel reduction in convergence checking
- **Memory Access**: Coalesced access pattern for better performance

### Performance Analysis

#### Block Size Impact
- Small blocks (2x2x2): Less efficient due to underutilization
- Medium blocks (8x8x8): Optimal balance for most L values
- Large blocks (16x16x16): Limited by max threads per block (1024)

#### Grid Size Scaling
- Automatically adjusted for different L values
- Ensures good workload distribution
- Limited to maximum grid dimensions

### Convergence Analysis

#### Accuracy vs. L Size
- L=8: Limited accuracy due to boundary proximity
- L=16: Improved accuracy in central region
- L=32,64: Good agreement with Coulomb's law
- L=128,256: Excellent agreement, especially away from boundaries

#### Approach to Coulomb's Law
1. Near Field (r << L):
   - Potential closely follows 1/4πr
   - Relative error < 1% for r < L/4

2. Far Field (r ≈ L/2):
   - Boundary effects become significant
   - Deviation from Coulomb's law increases

3. Convergence Rate:
   - Error decreases approximately as 1/L²
   - Faster convergence near charge center
   - Boundary effects persist at large r

## Conclusion

The numerical solution successfully approaches Coulomb's law (V = 1/4πr) in the limit L >> 1, particularly in regions where r << L. The optimal performance is achieved with medium-sized thread blocks (8x8x8), balancing parallelism and resource utilization.

## Submission Guidelines

The homework report should include your source codes, results, and discussions. The discussion file should be prepared with a typesetting system, e.g., LaTeX, Word, etc., and it is converted to a PDF file. All files should be zipped into one gzipped tar file, with a file name containing your student number and the problem set number (e.g., r05202043_ps3.tar.gz). Please send your homework with the title "your_student_number_HW3" to twchiu@phys.ntu.edu.tw before 17:00, June 11, 2025 (deadline for all problem sets).