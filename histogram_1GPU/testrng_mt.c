/*************
* testrng_mt *             
*************/

/* An example program of calling the gsl_rng_mt19937 */

/* 
   To compile, type the following command.

   gcc -O3 -o testrng_mt testrng_mt.c -lgsl -lgslcblas -lm

*/

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>   /* header file for using gsl_rng */

int main(void)    
{
  gsl_rng  *rng;            /* pointer to gsl_rng random number generator */
  int      i,ia,num;
  float    x,y,fnum;
  double   dnum,mean,sigma,mean1,sigma1;
  double   fx,dx,w;
  double   f(double);	    /* function for MC integration */ 
  char     ct;

  printf("Welcome to the class of Computational Physics\n");
  printf("---------------------------------------------\n");
  printf("Enter command (? for help)\n");
  while((ct=getchar()) != 'Q') {
    switch(ct) {
      case 'N':
        rng = gsl_rng_alloc(gsl_rng_mt19937);     /* allocate RNG to mt19937 */
        printf("Enter the seed:\n");
        scanf("%d",&ia);
        gsl_rng_set(rng,ia);                     /* set the seed */
        printf("The RNG has been initialized\n");
        printf("ready for commands\n");
        fflush(stdout);
        break;
      case 'S':
        printf("How many random numbers to be generated ?\n");
        scanf("%lf",&dnum);
        mean=0.0;
        sigma=0.0;
        printf("  mean        standard deviation\n");
        printf("--------------------------------\n");
        for(i=0;i<dnum;i++) {
          dx = gsl_rng_uniform(rng);   /* generate a random number with uniform deviate */
          mean += dx;
          sigma += dx*dx;
        }
        mean /= dnum;
        sigma=sqrt(sigma/dnum-mean*mean);
        printf("%10.7f           %10.7f\n",mean,sigma);
        printf("ready for commands\n");
        fflush(stdout);
        break;
      case 'I':
        printf("Monte-Carlo integration of one-dimensional integral\n");
        printf("How many random numbers to be generated ?\n");
        scanf("%lf",&dnum);
        mean=0.0;     /* for simple sampling */
        sigma=0.0;
        mean1=0.0;    /* for importance sampling */
        sigma1=0.0;
        for(i=0;i<dnum;i++) {
          y = gsl_rng_uniform(rng);   /* generate a random number with uniform deviate */
	  fx = f(y);
          mean += fx;
          sigma += fx*fx;
          x=2.0-sqrt(4.0-3.0*y);
          w=(4.0-2.0*x)/3.0;
          fx = f(x)/w;
          mean1 += fx;
          sigma1 += fx*fx;
        }
        mean /= dnum;
        sigma=sqrt((sigma/dnum-mean*mean)/dnum);
        mean1 /= dnum;
        sigma1 = sqrt((sigma1/dnum-mean1*mean1)/dnum);
        printf("Simple sampling:     %.10f +/- %.10f\n",mean,sigma);
        printf("Importance sampling: %.10f +/- %.10f\n",mean1,sigma1);
        printf("Exact solution:      %.10f \n",acos(-1)/4.0);    /*   pi/4   */
        printf("ready for commands\n");
        fflush(stdout);
        break;
      case '?':
        printf("Commands are:\n");
        printf("=======================================================\n");
        printf(" N : initialize the RNG\n");
        printf(" S : perform simple tests of the RNG\n");
        printf(" I : Monte-Carlo integration of one-dimensional integral\n");
        printf(" Q : quit \n");
        printf("-------------------------------------------------------\n");
        printf("ready for commands\n");
        fflush(stdout);
        break;
      case '\n':
        break;
      default:
        printf("unknown command\n");
        fflush(stdout);
        break;
    }
  }
  printf("Have a good day !\n"); 
  exit(0);
}



/* the function for MC integration */

double f(double x)		
{
   return(1/(1+x*x));
}

