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
    srand(seed);

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

    // Initialization of random integers
    srand(seed);

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

    // Get the number of chunks
    int num_chunks = num_runs / num_omp_threads_used;

    // Compute cumulative sum
    for (i=0; i<num_runs; i++)
	cumulative_sum = cumulative_sum + (double)pow((all_timings[i] - avg_time) / (double)(num_runs * num_omp_threads_used), 2);

    // Compute variance
    variance = cumulative_sum / (double)(num_chunks);

    // Now compute standard dev and return its value
    stdev = sqrt(variance) / (double)(num_chunks);
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
    int num_chunks = num_iterations / num_omp_threads;
    printf("  OpenMP threading info:\n");
    printf("    Using %s threads\n", num_set_omp_threads);
    printf("    Using %d chunks\n\n", num_chunks);

    //Print out alignment info (if any)
    printf("  Alignment info:\n");
    if (ALIGNMENT == 1)
        printf("    Unaligned\n");
    else
	printf("    %d bytes\n", ALIGNMENT);


#ifdef UNALIGNED

    // Set up matrix params
    int nonaligned_dim_m;
    int nonaligned_dim_n;
    int nonaligned_dim_k;
    int seed_omp_nonaligned;

    // Set up iteration params
    int num_nonaligned_iterations;

    // Print number of iterations
    printf("----------------------------------------------------\n");
    printf("Unaligned GOMP matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    // Begin
    #pragma omp parallel private(i, nonaligned_dim_m, nonaligned_dim_n, nonaligned_dim_k, seed_omp_nonaligned, num_nonaligned_iterations)
    {
        // Set values for m, n, and k
        nonaligned_dim_m = m;
	nonaligned_dim_n = n;
	nonaligned_dim_k = k;

        // Do the same with 'seed'
        seed_omp_nonaligned = seed;

	// Initialize elapsed time vars
	double nonaligned_elapsed_time = 0.0;
	double total_nonaligned_elapsed_time = 0.0;
	double avg_nonaligned_elapsed_time = 0.0;
	double nonaligned_standard_dev = 0.0;
	double *nonaligned_omp_run_timings;

	// Set up other time vars
        struct timespec start_time;
        struct timespec end_time;

	// Define number of nonaligned iterations
	num_nonaligned_iterations = num_iterations;

	// Do not use dynamic threading
        omp_set_dynamic(0);

	// Get number of threads
	double num_threads_used = (double)omp_get_num_threads();

	// Set up matrices
	double *omp_nonaligned_matrix_A = (double *)malloc(nonaligned_dim_m * nonaligned_dim_n * sizeof(double));
	double *omp_nonaligned_matrix_B = (double *)malloc(nonaligned_dim_n * nonaligned_dim_k * sizeof(double));
	double *omp_nonaligned_matrix_C = (double *)malloc(nonaligned_dim_m * nonaligned_dim_k * sizeof(double));

	// Setup array to hold all timing data
	nonaligned_omp_run_timings = (double*)malloc(num_nonaligned_iterations * sizeof(double));

	// Populate matrices A and B with data
        ompPopulateMatrix(omp_nonaligned_matrix_A, nonaligned_dim_m, nonaligned_dim_n, seed_omp_nonaligned);
        ompPopulateMatrix(omp_nonaligned_matrix_B, nonaligned_dim_n, nonaligned_dim_k, seed_omp_nonaligned);

	// Matrix multiply
	#pragma omp for
	for (i=0; i<num_nonaligned_iterations; i++){
            clock_gettime(CLOCK_REALTIME, &start_time);
            ompMatrixMultiply(omp_nonaligned_matrix_A, omp_nonaligned_matrix_B, omp_nonaligned_matrix_C, nonaligned_dim_m, nonaligned_dim_n, nonaligned_dim_k);
	    clock_gettime(CLOCK_REALTIME, &end_time);

	    // Compute elapsed time
	    nonaligned_elapsed_time = 0.0;
            nonaligned_elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
            nonaligned_elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
            nonaligned_elapsed_time /= 1000.0;
	    nonaligned_elapsed_time;
    
	    // Print out elapsed time for run #i
            printf("  Run #%d: %0.3f sec\n", i+1, nonaligned_elapsed_time / num_omp_threads);

	    // Keep track of total elapsed time
	    total_nonaligned_elapsed_time += nonaligned_elapsed_time;

	    // Hold onto the performance timings
	    nonaligned_omp_run_timings[i] = nonaligned_elapsed_time;

	    // Reset matrix C
	    resetMatrix(omp_nonaligned_matrix_C, nonaligned_dim_m, nonaligned_dim_k);
	}

        // Compute statistics
        #pragma omp single
	{
        // Update total runtime. (Each thread is capturing the performance timings, so we need to combine the total of all threads)
	avg_nonaligned_elapsed_time = total_nonaligned_elapsed_time / (double)num_nonaligned_iterations;
	nonaligned_standard_dev = computeStandardDev(nonaligned_omp_run_timings, num_nonaligned_iterations, avg_nonaligned_elapsed_time);

	// Print timings
	printf("  RUN_STATISTICS:\n");
	printf("  >> Total runtime : %0.3f sec\n", total_nonaligned_elapsed_time);
	double chunk_timing = 0.0;
	int chunk_count = 1;
        for (i=0; i<num_nonaligned_iterations; i++){

	    // Extract the time it took to run the chunk
	    chunk_timing = nonaligned_omp_run_timings[i];

	    // If the timing is greater than 0, then we want to print the chunk timing
	    if (isgreater((float)chunk_timing, 0.0f)){
                printf("        - Chunk %d took : %0.3f sec\n", chunk_count, chunk_timing);
		chunk_count++;
	    }

	    // If we've already gone through all of our chunks, then break the loop
	    if (chunk_count > num_chunks)
                break;
	}
	printf("  >> Average runtime overall (per run) : %0.2f +/- %0.2f sec\n", avg_nonaligned_elapsed_time, nonaligned_standard_dev);
	}
    }	

#endif

#ifdef ALIGNED
    // Initialize vars
    int i_omp_aligned;
    int seed_omp_aligned;
    int num_aligned_iterations;
    int aligned_dim_m;
    int aligned_dim_n;
    int aligned_dim_k;
    int alignment;

    // Print number of iterations
    printf("----------------------------------------------------\n");
    printf("Aligned GOMP matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    #pragma omp parallel shared(alignment, seed_omp_aligned, num_aligned_iterations, aligned_dim_m, aligned_dim_n, aligned_dim_k) private(i_omp_aligned)
    {
	// Do not use dynamic threading
        omp_set_dynamic(0);

	// Get number of threads
	double num_threads_used = (double)omp_get_num_threads();

        // Set seed
        seed_omp_aligned = seed;

	// Set number of iterations
	num_aligned_iterations = num_iterations;

	// Set dimensions
        aligned_dim_m = m;
	aligned_dim_n = n;
	aligned_dim_k = k;

	// Set alignment
	alignment = ALIGNMENT;

        // Set alignment dims
        size_t mat_A_aligned_size = ((size_t) (aligned_dim_m * aligned_dim_n * sizeof(double)) + alignment - 1) & (~(alignment - 1));
        size_t mat_B_aligned_size = ((size_t) (aligned_dim_n * aligned_dim_k * sizeof(double)) + alignment - 1) & (~(alignment - 1));
        size_t mat_C_aligned_size = ((size_t) (aligned_dim_m * aligned_dim_k * sizeof(double)) + alignment - 1) & (~(alignment - 1));

	// Initialize matrices
	double *omp_aligned_matrix_A = (double*)aligned_alloc(alignment, mat_A_aligned_size);
        double *omp_aligned_matrix_B = (double*)aligned_alloc(alignment, mat_B_aligned_size);
	double *omp_aligned_matrix_C = (double*)aligned_alloc(alignment, mat_C_aligned_size);

        ompPopulateMatrix(omp_aligned_matrix_A, aligned_dim_m, aligned_dim_n, seed_omp_aligned);
        ompPopulateMatrix(omp_aligned_matrix_B, aligned_dim_n, aligned_dim_k, seed_omp_aligned);

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
	aligned_omp_run_timings = (double*)malloc(num_aligned_iterations * sizeof(double));

	// Populate matrices A and B with data

	// Matrix multiply
        #pragma omp for
	for (i=0; i<num_aligned_iterations; i++){
            clock_gettime(CLOCK_REALTIME, &start_time);
            ompSIMDMatrixMultiply(omp_aligned_matrix_A, omp_aligned_matrix_B, omp_aligned_matrix_C, aligned_dim_m, aligned_dim_n, aligned_dim_k, alignment);
	    clock_gettime(CLOCK_REALTIME, &end_time);

	    // Compute elapsed time
	    aligned_elapsed_time = 0.0;
            aligned_elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000.0;
            aligned_elapsed_time += (end_time.tv_nsec - start_time.tv_nsec) / 1000000.0;
            aligned_elapsed_time /= 1000.0;
    
	    // Print out elapsed time for run #i
            printf("  Run #%d: %0.3f sec\n", i+1, aligned_elapsed_time / num_omp_threads);

	    // Keep track of total elapsed time
	    total_aligned_elapsed_time += aligned_elapsed_time;

	    // Hold onto the performance timings
	    aligned_omp_run_timings[i] = aligned_elapsed_time;
	    
	    // Reset matrix C
	    resetMatrix(omp_aligned_matrix_C, aligned_dim_m, aligned_dim_k);
	}

        // Compute statistics
        #pragma omp single
	{	
	avg_aligned_elapsed_time = total_aligned_elapsed_time / (double)num_aligned_iterations;
	aligned_standard_dev = computeStandardDev(aligned_omp_run_timings, num_aligned_iterations, avg_aligned_elapsed_time);

	// Print timings
	printf("  RUN_STATISTICS:\n");
	printf("  >> Total runtime : %0.3f sec\n", total_aligned_elapsed_time);
	double chunk_timing = 0.0;
	int chunk_count = 1;
        for (i=0; i<num_aligned_iterations; i++){

	    // Extract the time it took to run the chunk
	    chunk_timing = aligned_omp_run_timings[i];

	    // If the timing is greater than 0, then we want to print the chunk timing
	    if (isgreater((float)chunk_timing, 0.0f)){
                printf("        - Chunk %d took : %0.3f sec\n", chunk_count, chunk_timing);
		chunk_count++;
	    }

	    // If we've already gone through all of our chunks, then break the loop
	    if (chunk_count > num_chunks)
                break;
	}

	printf("  >> Average runtime overall (per run) : %0.2f +/- %0.2f sec\n", avg_aligned_elapsed_time, aligned_standard_dev);
	}
    }
#endif

    return 0;
}
