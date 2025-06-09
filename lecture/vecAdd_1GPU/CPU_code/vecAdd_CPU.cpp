// Vector addition: C = 1/A + 1/B.
// compile with the following command:
//
// g++ -O3 -o vecAdd_CPU vecAdd_CPU.cpp


// Includes
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <time.h>
//using namespace std;

// Set up the clock for timing
clock_t t_start, t_end;

// Variables
float* h_A;   // host vectors
float* h_B;
float* h_D;

// Functions
void RandomInit(float*, int);
void test_rand();
void test_rand_16807();
void srand_16807();
int rand_16807();

int seed;

// Host code

int main( )
{

    printf("Vector Addition: C = 1/A + 1/B\n");
    int mem = 1024*1024*1024;     // Giga    
    int N;

    int testr;
    int iseed;
    printf("To enter the seed for the initialization of RNG ? (1/0) ");
    scanf("%d",&testr);
    printf("%d\n",testr); 
    if(testr == 1) {
      printf("Enter the seed: ");
      scanf("%d",&iseed);
      printf("%d\n",iseed); 
    }
    else 
      seed = time(NULL);

    srand(iseed);
    seed = iseed;
    srand_16807();

    int testp;
    printf("No test / Test rand / Test rand_16870 ? (0/1/2) ");
    scanf("%d",&testp);
    printf("%d\n",testp);

    if(testp == 1) {
      printf("Testing rand ...\n");
      test_rand();
    }  
    else if(testp == 2) {
      printf("Testing rand_16807 ...\n");
      test_rand_16807();
    }  


next:
    printf("Enter the size of the vectors: ");
    scanf("%d",&N);        
    printf("%d\n",N);        
    if( 3*N > mem ) {     // each real number takes 4 bytes
      printf("The size of these 3 vectors cannot be fitted into 4 Gbyte\n");
      goto next;
    }
    long size = N * sizeof(float);


    // Allocate input vectors h_A and h_B in host memory

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);

    
    // Initialize input vectors

    RandomInit(h_A, N);
    RandomInit(h_B, N);

    h_D = (float*)malloc(size);    // the reference solution

    t_start = clock();   // start the clock 

    for (int i = 0; i < N; i++)
        h_D[i] = 1.0/h_A[i] + 1.0/h_B[i];

    t_end = clock();     // stop the clock

    double cputime = 1000*((double)(t_end - t_start))/CLOCKS_PER_SEC;  // milli-second
    printf("Processing time for CPU: %f (ms) \n",cputime);
    printf("CPU Gflops: %f\n",3*N/(1000000.0*cputime));

}


// Allocates an array with random float entries.
void RandomInit(float* data, int n)
{
    for (int i = 0; i < n; i++)
        data[i] = rand() / (float)RAND_MAX;
}


void test_rand()
{
   int x,y,z;
   double i;

   printf("RAND_MAX = %d\n", RAND_MAX);

   x = rand();
   y = rand();
   printf("x = %d\n", x);
   printf("y = %d\n", y);
   printf("To check whether the algorithm of rand() is x(i+1) = 16807*x(i) mod (2^31 -1)\n");
   z = ( (long) 16807*x)%RAND_MAX;
   if(z == y)  
     printf("z = %d = y, the algorithm of rand is same as above\n", z);
   else 
     printf("z = %d, z != y, the algorithm of rand is different from above !\n", z);

   i = 1.0;
   while ( y != x ) {
     if( (y == 0) || (y == RAND_MAX) ) {
       printf("y = %d, i = %f\n", y, i);
     }
     y = rand();
     i++;
   }

   printf("y = %d\n",y);
   printf("The period of rand() is %f\n",i);

}

void test_rand_16807()
{
   int x,y,z;
   double i;

   x = rand_16807();
   y = rand_16807();
   printf("x = %d\n", x);
   printf("y = %d\n", y);
   printf("To check whether the algorithm of rand_16807() is x(i+1) = 16807*x(i) mod (2^31 -1)\n");
   z = ( (long) 16807*x)%RAND_MAX;
   if(z == y)  
     printf("z = %d = y, the algorithm of rand_16807 is same as above\n", z);
   else 
     printf("z = %d, z != y, the algorithm of rand_16807 is different from above !\n", z);

   i = 1.0;
   while ( y != x ) {
     if( (y == 0) || (y == RAND_MAX) ) {
       printf("y = %d, i = %f\n", y, i);
     }
     y = rand_16807();
     i++;
   }

   printf("y = %d\n",y);
   printf("The period of rand_16807 is %f\n",i);

}

int rand_16807()
{
   static long int a = seed;

   a = 16807*a % 2147483647;
   return( (int) a); 
}


void srand_16807()
{
   if(seed == 0)
     seed += 137;
}

