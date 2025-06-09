/********
* metro *                 T.W. Chiu     Oct.2017
********/

/* An example program of calling the gsl_rng_mt19937 to perform Monte-Carlo integration */
/* with simple sampling, importance sampling, and Metropolis algorithm.                 */

/* 
   To compile, type the following command.

   g++ -o metro metro.c -lgsl -lgslcblas

*/

#include <stdio.h>
#include <math.h>
#include <gsl/gsl_rng.h>   /* header file for using gsl_rng */

int main(void)    
{
  gsl_rng  *rng;            /* pointer to gsl_rng random number generator */
  int      i,j,ia;
  double   x,y,r,x_new,x_old,delta;
  double   dnum,mp,mean,sigma,mean1,sigma1;
  double   fx,dx,ww,w_old,w_new,f_old,f_new;
  double   f(double);	    /* function for MC integration */ 
  double   w(double);	    /* weight function for MC integration */ 
  double   *seqx;           /* The array of {x_0, x_1, x_2, ..., x_(N-1)} */
  double   *part;           /* The array of N(z) for m partitions */
  double   zj;              /* center of the j-th partition */
  char     ct;

  rng = gsl_rng_alloc(gsl_rng_mt19937);     /* allocate RNG to mt19937 */
  gsl_rng_set(rng,12345);                   /* set the initial seed */

  printf("Welcome to the class of Computational Physics\n");
  printf("---------------------------------------------\n");
  printf("Enter command (? or help)\n");
  while((ct=getchar()) != 'Q') {
    switch(ct) {
      case 'N':
//        rng = gsl_rng_alloc(gsl_rng_mt19937);     /* allocate RNG to mt19937 */
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
          ww=(4.0-2.0*x)/3.0;
          fx = f(x)/ww;
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
      case 'M':
        printf("Monte-Carlo integration with Metropolis algorithm: w(x)=(4-2x)/3\n");
        printf("How many samples to be generated ?\n");
        scanf("%lf",&dnum);
        mean=0.0;    
        sigma=0.0;
        x_old = gsl_rng_uniform(rng);   /* generate a random number with uniform deviate */
    	w_old = w(x_old);
        for(i=0;i<dnum;i++) {
          x_new = gsl_rng_uniform(rng);   
    	  w_new = w(x_new);
    	  if(w_new >= w_old) {
            w_old = w_new;
	    x_old = x_new;
     	  }  
          else {
            r = gsl_rng_uniform(rng);   
            if(r <= w_new/w_old ) {
              w_old = w_new;
              x_old = x_new;
	    }
          }
          x = x_old;	  
          fx = f(x)/w(x); 	  
          mean += fx;
          sigma += fx*fx;
        }
        mean /= dnum;
        sigma=sqrt((sigma/dnum-mean*mean)/dnum);
        printf("Metropolis sampling: %.10f +/- %.10f\n",mean,sigma);
        printf("Exact solution:      %.10f \n",acos(-1)/4.0);    /*   pi/4   */
        printf("ready for commands\n");
        fflush(stdout);
        break; 
      case 'K':
        printf("Monte-Carlo integration with Metropolis algorithm: w(x)=f(x)/I and partitioning\n");
        printf("How many samples to be generated ?\n");
        scanf("%lf",&dnum);
        printf("How many partitions ?\n");
        scanf("%lf",&mp);
    	seqx = (double*)malloc(dnum*sizeof(double)); 
    	part = (double*)malloc(mp*sizeof(double));      /* array for N(z) */
    	for(i=0;i<mp;i++) { 
          part[i]=0.0;
    	}  
    	delta = 1.0/(double)mp;         /* width of a partition */
        x_old = gsl_rng_uniform(rng);   /* generate a random number with uniform deviate */
    	f_old = f(x_old);
        for(i=0;i<dnum;i++) {
          x_new = gsl_rng_uniform(rng);   
    	  f_new = f(x_new);
    	  if(f_new >= f_old) {
    	    x_old = x_new;
            f_old = f_new;
    	  }  
    	  else {
            r = gsl_rng_uniform(rng);  
            if(r <= (f_new/f_old) ) {
              x_old = x_new;
              f_old = f_new;
	        }
          }
          seqx[i]=x_old;  /* the array {x_0, x_1, ..., x_(N-1)} can be used for the determination of p(x) */
    	  j=(int)(x_old/delta);   /* to determine which partition x falls into */
	      part[j] +=1.0;
        }
        mean=0.0;    
        sigma=0.0;
    	for(j=0;j<mp;j++) {
          zj = (j+0.5)*delta;
    	  if(part[j]>0) {
            fx = dnum*f(zj)/(part[j]*mp);		
            mean += fx;		
            sigma += fx*fx;
     	  }  
	    }  
        mean /= mp;
        sigma = sqrt((sigma/mp-mean*mean)/mp);
        printf("Metropolis sampling: %.10f +/- %.10f\n",mean,sigma);
        printf("Exact solution:      %.10f \n",acos(-1)/4.0);    /*   pi/4   */
    	free(seqx);
    	free(part);
        printf("ready for commands\n");
        fflush(stdout);
        break;
      case 'T':
        printf("Monte-Carlo integration with Metropolis algorithm: w(x)=f(x)/I\n");
        printf("(This demonstrates that Metropolis algorithm with w(x)=f(x)/I does not work !)\n");
        printf("How many samples to be generated ?\n");
        scanf("%lf",&dnum);
        mean=0.0;    
        sigma=0.0;
        x_old = gsl_rng_uniform(rng);   /* generate a random number with uniform deviate */
    	f_old = f(x_old);
        for(i=0;i<dnum;i++) {
          x_new = gsl_rng_uniform(rng);   
    	  f_new = f(x_new);
    	  if(f_new >= f_old) {
            f_old = f_new;
	        x_old = x_new;
    	  }  
	  	  else {
            r = gsl_rng_uniform(rng);   
            if(r <= f_new/f_old ) {
              f_old = f_new;
              x_old = x_new;
	        }
          }
          fx = f(x_old); 	  
          mean += fx;
          sigma += fx*fx;
        }
        mean /= dnum;
        sigma=sqrt((sigma/dnum-mean*mean)/dnum);
        printf("Metropolis sampling: %.10f +/- %.10f\n",mean,sigma);
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
        printf(" M : Monte-Carlo integration with Metropolis algorithm: w(x)=(4-2x)/3\n");
//        printf(" K : Monte-Carlo integration with Metropolis algorithm: w(x)=f(x)/I\n");
//        printf(" T : Monte-Carlo integration with Metropolis algorithm: w(x)=f(x)/I'\n");
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

/* the weight function for MC integration */

double w(double x)		
{
   return((4-2*x)/3);
}
