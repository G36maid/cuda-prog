// Solve the Laplace equation on a 2D lattice with boundary conditions.
//
// compile with the following command:
//
//
// (for GTX1060)
// nvcc -arch=compute_61 -code=sm_61,sm_61 -O3 -m64 -o laplace laplace.cu


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// field variables
float* h_new;   // host field vectors
float* h_old;   

int     MAX=1000000;          // maximum iterations
double  eps=1.0e-10;          // stopping criterion

int main(void)
{
    printf("Solve Laplace equation on a 2D lattice with boundary conditions\n");

    int Nx,Ny;    // lattice size

    printf("Enter the size of the square lattice: ");
    scanf("%d %d",&Nx,&Ny);        
    printf("%d %d\n",Nx,Ny);        

    int size = Nx*Ny*sizeof(float); 
    h_new = (float*)malloc(size);
    h_old = (float*)malloc(size);

    memset(h_old, 0, size);
    memset(h_new, 0, size);

//    for(int j=0;j<Ny;j++) 
//    for(int i=0;i<Nx;i++) 
//      h_new[i+j*Nx]=0.0;

    // Initialize the field vector with boundary conditions

    for(int x=0; x<Nx; x++) {
      h_new[x+Nx*(Ny-1)]=1.0;  
      h_old[x+Nx*(Ny-1)]=1.0;
    }  

    FILE *out1;          // save initial configuration in phi_initial.dat
    out1 = fopen("phi_initial.dat","w");

    fprintf(out1, "Inital field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(out1,"%.2e ",h_new[i+j*Nx]);
      }
      fprintf(out1,"\n");
    }
    fclose(out1);

    printf("\n");
    printf("Inital field configuration:\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        printf("%.2e ",h_new[i+j*Nx]);
      }
      printf("\n");
    }
    printf("\n");

    // create the timer
    cudaEvent_t start,stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    //start the timer
    cudaEventRecord(start,0);

    // to compute the reference solution

    double error = 10*eps;  // any value bigger eps is OK 
    int iter = 0;           // counter for iterations

    volatile bool flag = true;     

    float t, l, r, b;     // top, left, right, bottom
    double diff; 
    int site, ym1, xm1, xp1, yp1;

    while ( (error > eps) && (iter < MAX) ) {
      if(flag) {
        error = 0.0;
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1) {   
          }
          else {
            site = x+y*Nx;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            b = h_old[ym1]; 
            l = h_old[xm1]; 
            r = h_old[xp1]; 
            t = h_old[yp1]; 
            h_new[site] = 0.25*(b+l+r+t);
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          }
        } 
        } 
      }
      else {
        error = 0.0;
        for(int y=0; y<Ny; y++) {
        for(int x=0; x<Nx; x++) { 
          if(x==0 || x==Nx-1 || y==0 || y==Ny-1) {
          }
          else {
            site = x+y*Nx;
            xm1 = site - 1;    // x-1
            xp1 = site + 1;    // x+1
            ym1 = site - Nx;   // y-1
            yp1 = site + Nx;   // y+1
            b = h_new[ym1]; 
            l = h_new[xm1]; 
            r = h_new[xp1]; 
            t = h_new[yp1]; 
            h_old[site] = 0.25*(b+l+r+t);
            diff = h_new[site]-h_old[site]; 
            error = error + diff*diff;
          } 
        }
        }
      }
      flag = !flag;
      iter++;
      error = sqrt(error);

//      printf("error = %.15e\n",error);
//      printf("iteration = %d\n",iter);

    }         // exit if error < eps

    printf("error = %.15e\n",error);
    printf("total iterations = %d\n",iter);
    
    // stop the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);

    float cputime;
    cudaEventElapsedTime( &cputime, start, stop);
    printf("Processing time for CPU: %f (ms) \n",cputime);
    double flops = 7.0*(Nx-2)*(Ny-2)*iter; 
    printf("CPU Gflops: %lf\n",flops/(1000000.0*cputime));

    // destroy the timer
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    FILE *outc;          // save final configuration in phi_CPU.dat
    outc = fopen("phi_CPU.dat","w");

    fprintf(outc,"Final field configuration (CPU):\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        fprintf(outc,"%.2e ",h_new[i+j*Nx]);
      }
      fprintf(outc,"\n");
    }
    fclose(outc);

    printf("\n");
    printf("Final field configuration (CPU):\n");
    for(int j=Ny-1;j>-1;j--) {
      for(int i=0; i<Nx; i++) {
        printf("%.2e ",h_new[i+j*Nx]);
      }
      printf("\n");
    }

    free(h_new);
    free(h_old);

}



