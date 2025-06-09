/* fft.cpp
 * 
 * This is a KISS implementation of
 * the Cooley-Tukey recursive FFT algorithm.
 * This works, and is visibly clear about what is happening where.
 *
 * To compile this with the GNU/GCC compiler:
 * g++ -o fft fft.cpp -lm
 *
 * To run the compiled version from a *nix command line:
 * ./fft
 *
 */

#include <complex>
#include <cstdio>
#include <cufft.h>

#define BATCH 1

#define M_PI 3.14159265358979323846 // Pi constant with double precision

using namespace std;

// separate even/odd elements to lower/upper halves of array respectively.
// Due to Butterfly combinations, this turns out to be the simplest way 
// to get the job done without clobbering the wrong elements.

void separate (complex<double>* a, int n) {
    complex<double>* b = new complex<double>[n/2];  // get temp heap storage
    for(int i=0; i<n/2; i++)    // copy all odd elements to heap storage
        b[i] = a[i*2+1];
    for(int i=0; i<n/2; i++)    // copy all even elements to lower-half of a[]
        a[i] = a[i*2];
    for(int i=0; i<n/2; i++)    // copy all odd (from heap) to upper-half of a[]
        a[i+n/2] = b[i];
    delete[] b;                 // delete heap storage
}

// N must be a power-of-2, or bad things will happen.
// Currently no check for this condition.
//
// N input samples in X[] are FFT'd and results left in X[].
// Because of Nyquist theorem, N samples means 
// only first N/2 FFT results in X[] are the answer.
// (upper half of X[] is a reflection with no new information).

void fft2 (complex<double>* X, int N) {
    if(N < 2) {
        // bottom of recursion.
        // Do nothing here, because already X[0] = x[0]
    } else {
        separate(X,N);      // all evens to lower half, all odds to upper half
        fft2(X,     N/2);   // recurse even items
        fft2(X+N/2, N/2);   // recurse odd  items
        // combine results of two half recursions
        for(int k=0; k<N/2; k++) {
            complex<double> e = X[k    ];   // even
            complex<double> o = X[k+N/2];   // odd
                         // w is the complex root of unity, w^N = 1
            complex<double> w = exp( complex<double>(0,-2.*M_PI*k/N) );
            X[k    ] = e + w * o;
            X[k+N/2] = e - w * o;
        }
    }
}

// simple test program
int main ()
{
    int nSamples;
    double nSeconds = 1.0;                       // total time for sampling
    complex<double> *x;                          // storage for sample data
    complex<double> *X1, *X2;                    // storage for FFT answer
    const int nFreqs = 5;
    double freq[nFreqs] = { 2, 5, 11, 17, 29 }; // known freqs for testing

    int gid;
    printf("Enter the GPU ID (0/1): ");
    scanf("%d",&gid);
    printf("%d\n", gid);
    cudaSetDevice(gid);
    printf("Enter the vector size: ");
    scanf("%d",&nSamples);
    printf("%d\n", nSamples);
    printf("Print the data (0/1) ? ");
    int io;
    scanf("%d",&io);
    printf("%d\n", io);

    double sampleRate = nSamples / nSeconds;     // n Hz = n / second 
    double freqResolution = sampleRate/nSamples; // freq step in FFT result

    x  = (complex<double> *)malloc(sizeof(complex<double>) * nSamples);
    X1 = (complex<double> *)malloc(sizeof(complex<double>) * nSamples);
    X2 = (complex<double> *)malloc(sizeof(complex<double>) * nSamples);

    // generate samples for testing
    for(int i=0; i<nSamples; i++) {
        x[i] = complex<double>(0.,0.);
        // sum several known sinusoids into x[]
        for(int j=0; j<nFreqs; j++) 
            x[i] += sin( 2*M_PI*freq[j]*i/nSamples );
        X1[i] = x[i];        // copy into X[] for FFT work & result
    }

    float  gputime, cputime;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cufftHandle         plan;
    cufftDoubleComplex *data;
    cudaMalloc((void**)&data, sizeof(cufftDoubleComplex)*nSamples*BATCH);
    cudaMemcpy(data, x, sizeof(double)*nSamples*2, cudaMemcpyHostToDevice);

    cudaEventRecord(start,0);
    if (cufftPlan1d(&plan, nSamples, CUFFT_Z2Z, BATCH) != CUFFT_SUCCESS) {
        fprintf(stderr, "CUFFT error: Plan creation failed.\n");
	exit(1);
    }
    if (cufftExecZ2Z(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
	fprintf(stderr, "CUFFT error: ExecZ2Z forward failed.\n");
	exit(1);
    }
    cudaMemcpy(X2, data, sizeof(double)*nSamples*2, cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gputime, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Use CPU to compute fft for this data
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start,0);
    fft2(X1,nSamples);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cputime, start, stop);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    complex<double> diff;
    double sum=0.0;

    if(io == 1) {
      printf("  n\tx[]\tX[]\tf\n");       // header line
      for(int i=0; i<nSamples; i++) {
        printf("% 3d\t%+.3f\t%+.3f\t%g\n",
               i, x[i].real(), abs(X1[i]), i*freqResolution );
      }
    }

    for(int i=0; i<nSamples; i++) {
      diff = X1[i] - X2[i];
      sum  = sum + abs(diff * conj(diff));
    }
    printf("Difference of solutions between CPU and GPU: %E\n", sum);

    printf("GPU time: %f (ms)\n", gputime);
    printf("CPU time: %f (ms)\n", cputime);
    printf("GPU speed up: %f \n", cputime/gputime);

    cufftDestroy(plan);
    cudaFree(data);
}

// eof
