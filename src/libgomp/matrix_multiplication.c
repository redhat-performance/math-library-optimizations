/***********************************************************
 * FILENAME: matrix_multiplication.c                       *
 *                                                         *
 * FILE DESCRIPTION: Code to execute matrix multiplication *
 *                   tests with and without OpenMP         *
 *                                                         *
 * AUTHOR: Courtney Pacheco <cpacheco@redhat.com>          *
 ***********************************************************/


#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <time.h>
#include <assert.h>
#include <malloc.h>
#include <ctype.h>
#include <unistd.h>

#define MAX_MAT_VALUE 100000
#define CHUNK 10
#define DEFAULT_SEED 5

void printMatrix(double *matrix, int rows, int cols){
/* Test function to print out a matrix
 *
 * Inputs
 * ------
 * matrix: *double
 *     Matrix (as a 1D array) to print out
 *
 * rows: int
 *     Number of rows in the matrix
 *
 * cols: int
 *     Number of columns in the matrix
 */

    int i, j;
    double mat_value;

    for (i=0; i<rows; i++){
        for (j=0; j<cols; j++){
	    mat_value = matrix[j + cols * i];
            printf("%0.1f\t", mat_value);
	    if (j == cols - 1){
                printf("\n");
	    }
	}
    }

};

void populateMatrix(double *matrix, int rows, int cols, int seed){
/* Populates a matrix with random values 
 *
 * Inputs
 * ------
 * **matrix: double
 *     The matrix to populate
 *
 * m: int
 *     The length of the matrix **matrix
 *
 * n: int
 *     The width of the matrix **matrix
 *
 * seed: int
 *     Random seed or default seed
 */

    // Set random double
    double random_double;

    // Initialization of random integers
    srand((unsigned int)seed);

    // Populate matrix
    int i, j;
    for (i=0; i<rows; i++){
        for (j=0; j<cols; j++){

            // Generate random int
            random_double = (double)rand() / MAX_MAT_VALUE;
            
	    // Store in matrix
	    matrix[i + j*rows] = random_double;
	}
    }
}

void ompPopulateMatrix(double *matrix, int rows, int cols, int seed){
/* Populates a matrix with random values with OpenMP 
 *
 * Inputs
 * ------
 * *matrix_as_array: double
 *     The matrix to populate, in the form of a single array
 *
 * m: int
 *     The length of the matrix **matrix
 *
 * n: int
 *     The width of the matrix **matrix
 *
 * seed: int
 *     Random seed or default seed
 */

#pragma omp parallel private(matrix, rows, cols)
{
    // Do not use dynamic threading
    omp_set_dynamic(0);

    // Initialization of random integers
    srand((unsigned int)seed);

    // Iterative vars
    int i, j;

    // Populate matrix
    #pragma omp for 
    for (i=0; i<rows; i++){
        for (j=0; j<cols; j++){
	    matrix[j + cols * i] = (double)rand() / (double)MAX_MAT_VALUE;
	}
    }
}
}

void matrixMultiply(double *mat_A, double *mat_B, double *mat_C, int m, int n, int k){
/* Multiplies matrices 'mat_A' and 'mat_B' to generate 'mat_C'
 *
 * Inputs
 * ------
 * mat_A: double*
 *     The first matrix, of size M x N
 *
 * mat_B: double*
 *     The second matrix, of size N x K
 *
 * mat_C: double*
 *     The resulting matrix
 *
 * m: int
 *     The number of rows in mat_A
 *
 * n: int 
 *     The number of cols in mat_A *or* number of rows in mat_B
 *
 * k: int 
 *     The number of cols in mat_B
 */
    // Iterative vars
    int i, j, h;

    // Temporary index vars
    int idx_A, idx_B, idx_C;

    // Temporary matrix val for matrix C
    double mat_value = 0.0;

    for (i=0; i<m; i++){

	// Set matrix value equal to zero to initialize each iteration
        for (j=0; j<k; j++){

	    // Index into matrix C
	    idx_C = j + i * m;

	    // Temporary value
	    mat_value = 0.0;

	    // Compute the matrix multiplication
            for (h=0; h<n; h++){
		idx_A = h + i * n;
                idx_B = j + h * k;
                mat_value += mat_A[idx_A] * mat_B[idx_B];
	    }

	    // Populate the matrix
	    mat_C[idx_C] = mat_value;
	}
    }
};

#pragma omp declare simd aligned(mat_A,mat_B,mat_C:ALIGNMENT)
void ompSIMDMatrixMultiply(double *__restrict__ mat_A, double *__restrict__ mat_B, double *__restrict__ mat_C, int m, int n, int k, int alignment){	
/* Uses OpenMP to multiply 'mat_A' and 'mat_B' to generate 'mat_C'
 *
 * Inputs
 * ------
 * mat_A: double**
 *     The first matrix, of size M x N
 *
 * mat_B: double**
 *     The second matrix, of size N x K
 *
 * mat_C: double**
 *     The resulting matrix
 *
 * m: int
 *     The number of rows in mat_A
 *
 * n: int 
 *     The number of cols in mat_A *or* number of rows in mat_B
 *
 * k: int 
 *     The number of cols in mat_B
 */

#pragma omp parallel shared(mat_A, mat_B, mat_C, m, n, k, alignment)
{
    // Iterative vars
    int i, j, h;

    // Temporary index vars
    int idx_A, idx_B, idx_C;

    // Temporary matrix val for matrix C
    double mat_value = 0.0;

    #pragma omp for
    for (i=0; i<m; i++){

	// Set matrix value equal to zero to initialize each iteration
        for (j=0; j<k; j++){

	    // Index into matrix C
	    idx_C = j + i * m;

	    // Temporary value
	    mat_value = 0.0;

	    // Compute the matrix multiplication
            for (h=0; h<n; h++){
		idx_A = h + i * n;
                idx_B = j + h * k;
                mat_value += mat_A[idx_A] * mat_B[idx_B];
	    }

	    // Populate the matrix
	    mat_C[idx_C] = mat_value;
	}
    }
}
};

void ompMatrixMultiply(double *mat_A, double *mat_B, double *mat_C, int m, int n, int k){	
/* Uses OpenMP to multiply 'mat_A' and 'mat_B' to generate 'mat_C'
 *
 * Inputs
 * ------
 * mat_A: double*
 *     The first matrix, of size M x N
 *
 * mat_B: double*
 *     The second matrix, of size N x K
 *
 * mat_C: double*
 *     The resulting matrix
 *
 * m: int
 *     The number of rows in mat_A
 *
 * n: int 
 *     The number of cols in mat_A *or* number of rows in mat_B
 *
 * k: int 
 *     The number of cols in mat_B
 */

#pragma omp parallel shared(mat_A, mat_B, mat_C, m, n, k)
{
    // Iterative vars
    int i, j, h;

    // Temporary index vars
    int idx_A, idx_B, idx_C;

    // Temporary matrix val for matrix C
    double mat_value = 0.0;

    #pragma omp for
    for (i=0; i<m; i++){

	// Set matrix value equal to zero to initialize each iteration
        for (j=0; j<k; j++){

	    // Index into matrix C
	    idx_C = i + j*k;

	    // Temporary value
	    mat_value = 0.0;

	    // Compute the matrix multiplication
            for (h=0; h<n; h++){
                idx_A = i + h * n;
		idx_B = h + j * k;
                mat_value += mat_A[idx_A] * mat_B[idx_B];
	    }

	    // Populate the matrix
	    mat_C[idx_C] = mat_value;
	}
    }
}
};

double resetMatrix(double *matrix_to_reset, int width, int height){
/* Resets a matrix by setting all its values to zero.
 *
 * Inputs
 * ------
 * matrix_to_reset: double**
 *    Self explanatory
 *
 * width: int
 *    Width of the matrix
 *
 * height: int
 *    Height of the matrix
 */

    // Initialize iteration vars
    int row;
    int col;

    // Clear matrix
    #pragma omp for collapse(2)
    for (row=0; row<width; row++){
        for (col=0; col<height; col++){
	    matrix_to_reset[col*width + row] = 0.0;
	}
    }
};

double computeStandardDev(double *all_timings, int num_runs, double avg_time){
/* Given a set of performance times, this function computes the standard dev across all runs
 *
 * Inputs
 * ------
 * all_timings: double*
 *     Array which holds all the performance times gathered.
 *
 * num_runs: int
 *     Number of runs captured during the experiment.
 *
 * avg_time: double
 *     Average performance time across all runs in 'all_timings'
 */

    // Initialize vars for computing standard dev
    double cumulative_sum = 0.0;
    double variance = 0.0;
    double stdev = 0.0;

    // Initialize iteration var
    int i;

    // Get number of threads being used
    double num_omp_threads_used = (double)omp_get_num_threads();

    // Compute cumulative sum
    for (i=0; i<num_runs; i++)
	cumulative_sum = cumulative_sum + (double)pow((all_timings[i] - avg_time) / (double)(num_runs * num_omp_threads_used), 2);

    // Compute variance
    variance = cumulative_sum / (double)(num_runs);

    // Now compute standard dev and return its value
    stdev = sqrt(variance);
    return stdev;
};

int isPositiveInteger(char value_to_check[]) { 
/* Checks if a string (char array) is actually a positive integer
 *
 * Inputs
 * ------
 * value_to_check: char[]
 *     User input to check.
 *
 * Returns
 * -------
 * 1 if true, 0 if false.
 */

    // Set iteration var
    int i;

    // Check if the first value is a zero. We don't want zeros.
    if (value_to_check[0] == '0')
	return 0;

    // Iterate through the char array to see if the value has any letters, hyphens (e.g., if the user
    // passed in '-100'), or other special chars.
    for (i=0; value_to_check[i] != 0; i++)
        if (!isdigit(value_to_check[i]))
            return 0;

    return 1;
} 


int main(int argc, char *argv[]){

    // Initialize vars
    int m, n, k, i, j, seed, num_iterations;

    // Make sure the user passed in 'm', 'n', and 'k'
    if (argc < 4){
        printf("\nIncomplete set of arguments. Required arguments: 'm', 'n', and 'k'. Optional arguments: 'num_iterations' and 'seed'.\n");
        exit(1);
    }
    else if (argc > 6){
        printf("\nToo many arguments. Required arguments: 'm', 'n', and 'k'. Optional arguments: (1.) num iterations, and (2.) 'seed'\n");
        exit(1);
    }

    // Check if each value in argv is an integer
    int arg_is_positive_integer;
    for (j=1; j<argc; j++){

	// Check if the given argument is a positive integer.
        arg_is_positive_integer = isPositiveInteger(argv[j]);

	// If the value is not a positive integer, then throw an error
	if (arg_is_positive_integer == 0){
            printf("ERROR: Argument #%d is not a positive integer. Each argument must be an integer > 0.\n", j);
	    exit(1);
	}
    }

    // If we've confirmed that the inputs are positive integers, let's parse them
    char *p;
    m = strtol(argv[1], &p, 10);
    n = strtol(argv[2], &p, 10);
    k = strtol(argv[3], &p, 10);

    // If the user passed in a number of iterations, then let's parse it
    if (argc == 5 || argc == 6)
        num_iterations = strtol(argv[4], &p, 10);
    else {
	num_iterations = sysconf(_SC_NPROCESSORS_ONLN);
    }

    // If the user passed in a seed, let's parse it
    if (argc == 6)
	seed = strtol(argv[5], &p, 10);
    else
        seed = DEFAULT_SEED;

    // Print header
    printf("---------------------\n");
    printf("PARSED INPUTS\n");
    printf("---------------------\n");

    // Print matrix info (MxN matrix)
    printf("  Matrix info:\n");
    printf("    Matrix A:\n");
    printf("        M=%d\n", m);
    printf("        N=%d\n", n);
    printf("    Matrix B:\n");
    printf("        N=%d\n", n);
    printf("        K=%d\n\n", k);

    // Print iteration info
    printf("  Iteration info:\n");
    if (argc == 5 || argc == 6)
        printf("    Using # of iterations = %d\n\n", num_iterations);
    else
        printf("    Using predefined # of iterations = %d (=%d cores)\n\n", num_iterations, num_iterations);

    // Print seed info
    printf("  Seed info:\n");
    if (argc == 6)
        printf("    Using seed = %d\n\n", seed);
    else
        printf("    Using predefined seed = DEFAULT_SEED = %d\n\n", seed);

    // Print number of threads and chunk info
    const char *num_set_omp_threads = getenv("OMP_NUM_THREADS");
    int num_omp_threads = strtol(num_set_omp_threads, &p, 10);
    printf("  OpenMP threading info:\n");
    printf("    Using %d threads\n\n", num_omp_threads);

    //Print out alignment info (if any)
    printf("  Alignment info:\n");
    if (ALIGNMENT == 1)
        printf("    Unaligned\n\n");
    else
	printf("    %d bytes\n\n", ALIGNMENT);


#ifdef UNALIGNED

    // Print number of iterations
    printf("----------------------------------------------------\n");
    printf("Unaligned GOMP matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    // Initialize elapsed time vars
    double nonaligned_elapsed_time = 0.0;
    double total_nonaligned_elapsed_time = 0.0;
    double avg_nonaligned_elapsed_time = 0.0;
    double nonaligned_standard_dev = 0.0;
    double *nonaligned_omp_run_timings;

    // Set up other time vars
    struct timespec start_time;
    struct timespec end_time;

    // Get number of threads
    double num_threads_used = (double)omp_get_num_threads();

    // Set up matrices
    double *omp_nonaligned_matrix_A = (double *)malloc(m * n * sizeof(double));
    double *omp_nonaligned_matrix_B = (double *)malloc(n * k * sizeof(double));
    double *omp_nonaligned_matrix_C = (double *)malloc(m * k * sizeof(double));

    // Setup array to hold all timing data
    nonaligned_omp_run_timings = (double*)malloc(num_iterations * sizeof(double));

    // Populate matrices A and B with data
    ompPopulateMatrix(omp_nonaligned_matrix_A, m, n, seed);
    ompPopulateMatrix(omp_nonaligned_matrix_B, n, k, seed);

    // Matrix multiply
    for (i=0; i<num_iterations; i++){
        clock_gettime(CLOCK_REALTIME, &start_time);
        ompMatrixMultiply(omp_nonaligned_matrix_A, omp_nonaligned_matrix_B, omp_nonaligned_matrix_C, m, n, k);
        clock_gettime(CLOCK_REALTIME, &end_time);

        // Compute elapsed time
        nonaligned_elapsed_time = 0.0;
        nonaligned_elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
        nonaligned_elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
        nonaligned_elapsed_time /= 1000.0;
        nonaligned_elapsed_time;
    
        // Print out elapsed time for run #i
        printf("  Run #%d:\t%0.3f sec\n", i+1, nonaligned_elapsed_time);

	// Keep track of total elapsed time
	total_nonaligned_elapsed_time += nonaligned_elapsed_time;

	// Hold onto the performance timings
	nonaligned_omp_run_timings[i] = nonaligned_elapsed_time;

	// Reset matrix C
	resetMatrix(omp_nonaligned_matrix_C, m, k);
    }

    // Compute statistics
    avg_nonaligned_elapsed_time = total_nonaligned_elapsed_time / (double)num_iterations;
    nonaligned_standard_dev = computeStandardDev(nonaligned_omp_run_timings, num_iterations, avg_nonaligned_elapsed_time);

    // Print timings
    printf("  RUN_STATISTICS:\n");
    printf("  >> Total runtime : %0.3f sec\n", total_nonaligned_elapsed_time);
    printf("  >> Average runtime overall (per run) : %0.2f +/- %0.2f sec\n", avg_nonaligned_elapsed_time, nonaligned_standard_dev);

#endif

#ifdef ALIGNED

    // Print number of iterations
    printf("----------------------------------------------------\n");
    printf("Aligned GOMP matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    // Get number of threads
    double num_threads_used = (double)omp_get_num_threads();

    // Set alignment dims
    size_t mat_A_aligned_size = ((size_t) (m * n * sizeof(double)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));
    size_t mat_B_aligned_size = ((size_t) (n * k * sizeof(double)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));
    size_t mat_C_aligned_size = ((size_t) (m * k * sizeof(double)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));

    // Initialize matrices
    double *omp_aligned_matrix_A = (double*)aligned_alloc(ALIGNMENT, mat_A_aligned_size);
    double *omp_aligned_matrix_B = (double*)aligned_alloc(ALIGNMENT, mat_B_aligned_size);
    double *omp_aligned_matrix_C = (double*)aligned_alloc(ALIGNMENT, mat_C_aligned_size);

    // Populate the matrices with values
    ompPopulateMatrix(omp_aligned_matrix_A, m, n, seed);
    ompPopulateMatrix(omp_aligned_matrix_B, n, k, seed);

    // Initialize elapsed time vars
    double aligned_elapsed_time = 0.0;
    double total_aligned_elapsed_time = 0.0;

    double avg_aligned_elapsed_time = 0.0;
    double aligned_standard_dev = 0.0;
    double *aligned_omp_run_timings;

    // Set up other time vars
    struct timespec start_time;
    struct timespec end_time;

    // Setup array to hold all timing data
    aligned_omp_run_timings = (double*)malloc(num_iterations * sizeof(double));

    // Matrix multiply
    for (i=0; i<num_iterations; i++){
		
        clock_gettime(CLOCK_REALTIME, &start_time);
        ompSIMDMatrixMultiply(omp_aligned_matrix_A, omp_aligned_matrix_B, omp_aligned_matrix_C, m, n, k, ALIGNMENT);
	clock_gettime(CLOCK_REALTIME, &end_time);

        // Compute elapsed time
        aligned_elapsed_time = 0.0;
        aligned_elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
        aligned_elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
        aligned_elapsed_time /= 1000.0;
    
        // Print out elapsed time for run #i
        printf("  Run #%d:\t%0.3f sec\n", i+1, aligned_elapsed_time);

        // Keep track of total elapsed time
        total_aligned_elapsed_time += aligned_elapsed_time;

        // Hold onto the performance timings
        aligned_omp_run_timings[i] = aligned_elapsed_time;
	    
         // Reset matrix C
        resetMatrix(omp_aligned_matrix_C, m, k);
    }

    // Compute statistics
    avg_aligned_elapsed_time = total_aligned_elapsed_time / (double)num_iterations;
    aligned_standard_dev = computeStandardDev(aligned_omp_run_timings, num_iterations, avg_aligned_elapsed_time);

    // Print timings
    printf("  RUN_STATISTICS:\n");
    printf("  >> Total runtime : %0.3f sec\n", total_aligned_elapsed_time);
    printf("  >> Average runtime overall (per run) : %0.2f +/- %0.2f sec\n", avg_aligned_elapsed_time, aligned_standard_dev);

#endif

#ifdef NO_OPENMP
    printf("----------------------------------------------------\n");
    printf("*No GOMP* matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    // Initialize matrices
    double *no_omp_matrix_A = (double *)malloc(m * n * sizeof(double));
    double *no_omp_matrix_B = (double *)malloc(n * k * sizeof(double));
    double *no_omp_matrix_C = (double *)malloc(m * k * sizeof(double));

    // Populate matrices with data
    populateMatrix(no_omp_matrix_A, m, n, seed);
    populateMatrix(no_omp_matrix_B, n, k, seed);

    // Initialize elapsed time vars
    double no_omp_elapsed_time = 0.0;
    double no_omp_total_elapsed_time = 0.0;
    double avg_no_omp_elapsed_time = 0.0;
    double no_omp_standard_dev = 0.0;
    double *no_omp_run_timings;

    // Set up other time vars
    struct timespec start_time;
    struct timespec end_time;

    // Setup array to hold all timing data
    no_omp_run_timings = (double*)malloc(num_iterations * sizeof(double));

    // Matrix multiply
    for (i=0; i<num_iterations; i++){
		
        clock_gettime(CLOCK_REALTIME, &start_time);
        matrixMultiply(no_omp_matrix_A, no_omp_matrix_B, no_omp_matrix_C, m, n, k);
	clock_gettime(CLOCK_REALTIME, &end_time);

        // Compute elapsed time
        no_omp_elapsed_time = 0.0;
        no_omp_elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
        no_omp_elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
        no_omp_elapsed_time /= 1000.0;
    
        // Print out elapsed time for run #i
        printf("  Run #%d:\t%0.3f sec\n", i+1, no_omp_elapsed_time);

        // Keep track of total elapsed time
        no_omp_total_elapsed_time += no_omp_elapsed_time;

        // Hold onto the performance timings
        no_omp_run_timings[i] = no_omp_elapsed_time;
	    
         // Reset matrix C
        resetMatrix(no_omp_matrix_C, m, k);
    }

    // Compute statistics
    avg_no_omp_elapsed_time = no_omp_total_elapsed_time / (double)num_iterations;
    no_omp_standard_dev = computeStandardDev(no_omp_run_timings, num_iterations, avg_no_omp_elapsed_time);

    // Print timings
    printf("  RUN_STATISTICS:\n");
    printf("  >> Total runtime : %0.3f sec\n", no_omp_total_elapsed_time);
    printf("  >> Average runtime overall (per run) : %0.2f +/- %0.2f sec\n", avg_no_omp_elapsed_time, no_omp_standard_dev);

#endif

    return 0;
}
