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

void populateMatrix(double **matrix, int m, int n, int seed){
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
    int row, col;
    for (row=0; row<m; row++){
        for (col=0; col<n; col++){

            // Generate random int
            random_double = (double)rand() / MAX_MAT_VALUE;
            
	    // Store in matrix
	    matrix[row][col] = random_double;
	}
    }
}

void ompPopulateMatrix(double **matrix, int *n_rows, int *n_cols, int seed){
/* Populates a matrix with random values with OpenMP 
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
    int row, col;
#pragma omp for schedule (static, CHUNK) 
    for (row=0; row<*n_rows; row++){
        for (col=0; col<*n_cols; col++){
	    // Store in matrix
	    matrix[row][col] = (double)rand() / (double)MAX_MAT_VALUE;
	}
    }
}

void matrixMultiply(double **mat_A, double **mat_B, double **mat_C, int m, int n, int k){
/* Multiplies matrices 'mat_A' and 'mat_B' to generate 'mat_C'
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

    // Iterative variables
    int h, i, j;

    // Placeholder
    double tmp;

    for (h=0; h<n; h++){

        for (i=0; i<m; i++){
            tmp = mat_A[i][h];

            for (j=0; j<k; j++){
                mat_C[h][j] += (tmp * mat_B[h][j]);
	    }
	}
    }
};

#pragma omp declare simd aligned(mat_A,mat_B,mat_C:ALIGNMENT)
void ompSIMDMatrixMultiply(double **__restrict__ mat_A, double **__restrict__ mat_B, double **__restrict__ mat_C, int m, int n, int k){	
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
    int a, b, c;

    #pragma omp for simd aligned(mat_A,mat_B,mat_C:ALIGNMENT)
    for (a=0; a<n; a++){

        for (b=0; b<m; b++){
	    mat_C[a][b] = 0.0;

            for (c=0; c<k; c++){
                mat_C[a][b] = (mat_C[a][b]) + (mat_A[a][c] * mat_B[c][b]);
	    }
	}
    }
};


void ompMatrixMultiply(double **mat_A, double **mat_B, double **mat_C, int m, int n, int k){	
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
    int a, b, c;

# pragma omp for schedule (static, CHUNK)
    for (a=0; a<n; a++){

        for (b=0; b<m; b++){
	    mat_C[a][b] = 0.0;

            for (c=0; c<k; c++){
                mat_C[a][b] = (mat_C[a][b]) + (mat_A[a][c] * mat_B[c][b]);
	    }
	}
    }
};

double resetMatrix(double **matrix_to_reset, int width, int height){
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
	    matrix_to_reset[row][col] = 0.0;
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

    //test
    printf("ALIGNMENT = %d\n", ALIGNMENT);


#ifdef UNALIGNED

    // Set up matrix params
    int m_omp_nonaligned;
    int n_omp_nonaligned;
    int k_omp_nonaligned;
    int seed_omp_nonaligned;

    // Set up iteration params
    int num_nonaligned_iterations;

    // Print number of iterations
    printf("----------------------------------------------------\n");
    printf("Unaligned GOMP matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    // Begin
    #pragma omp parallel private(i, m_omp_nonaligned, n_omp_nonaligned, k_omp_nonaligned, seed_omp_nonaligned, num_nonaligned_iterations)
    {
        // Set values for m, n, and k
        m_omp_nonaligned = m;
	n_omp_nonaligned = n;
	k_omp_nonaligned = k;

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
	double **omp_nonaligned_matrix_A = (double **)malloc(m_omp_nonaligned * sizeof(double*));
	double **omp_nonaligned_matrix_B = (double **)malloc(n_omp_nonaligned * sizeof(double*));
	double **omp_nonaligned_matrix_C = (double **)malloc(m_omp_nonaligned * sizeof(double*));
	for (i=0; i<m; i++){
            omp_nonaligned_matrix_A[i] = (double *)malloc(n_omp_nonaligned * sizeof(double));
            omp_nonaligned_matrix_C[i] = (double *)malloc(k_omp_nonaligned * sizeof(double));
        }

	for (i=0; i<n; i++)
            omp_nonaligned_matrix_B[i] = (double *)malloc(k_omp_nonaligned * sizeof(double));

	// Setup array to hold all timing data
	nonaligned_omp_run_timings = (double*)malloc(num_nonaligned_iterations * sizeof(double));

	// Populate matrices A and B with data
        ompPopulateMatrix(omp_nonaligned_matrix_A, &m_omp_nonaligned, &n_omp_nonaligned, seed_omp_nonaligned);
        ompPopulateMatrix(omp_nonaligned_matrix_B, &n_omp_nonaligned, &k_omp_nonaligned, seed_omp_nonaligned);

	// Matrix multiply
	#pragma omp for
	for (i=0; i<num_nonaligned_iterations; i++){
            clock_gettime(CLOCK_REALTIME, &start_time);
            ompMatrixMultiply(omp_nonaligned_matrix_A, omp_nonaligned_matrix_B, omp_nonaligned_matrix_C, m_omp_nonaligned, n_omp_nonaligned, k_omp_nonaligned);
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
	    resetMatrix(omp_nonaligned_matrix_C, m_omp_nonaligned, k_omp_nonaligned);
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
    int m_omp_aligned;
    int n_omp_aligned;
    int k_omp_aligned;
    int seed_omp_aligned;
    int num_aligned_iterations;
    size_t n_col;
    size_t k_col;
    size_t m_row;
    size_t n_row;
    size_t alignment;

    // Set aligned sizes
    size_t m_aligned_row_size = ((size_t) (m * sizeof(double*)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));;
    size_t n_aligned_row_size = ((size_t) (n * sizeof(double*)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));;
    size_t n_aligned_col_size = ((size_t) (n * sizeof(double)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));;
    size_t k_aligned_col_size = ((size_t) (k * sizeof(double)) + ALIGNMENT - 1) & (~(ALIGNMENT - 1));

    // Print number of iterations
    printf("----------------------------------------------------\n");
    printf("Aligned GOMP matrix multiplication across %d runs:\n", num_iterations);
    printf("----------------------------------------------------\n");

    #pragma omp parallel shared(m_row, n_col, n_row, k_col, alignment, seed_omp_aligned, num_aligned_iterations) private(i_omp_aligned)
    {
	// Do not use dynamic threading
        omp_set_dynamic(0);

	// Get number of threads
	double num_threads_used = (double)omp_get_num_threads();

	// Set dimension vars
	m_omp_aligned = m;
	n_omp_aligned = n;
	k_omp_aligned = k;

        // Set seed
        seed_omp_aligned = seed;

        // Set aligned row and col sizes
	m_row = m_aligned_row_size;
	n_row = n_aligned_row_size;
	n_col = n_aligned_col_size;
	k_col = k_aligned_col_size;

	// Set alignment within this block
	alignment = ALIGNMENT;

	// Set number of iterations
	num_aligned_iterations = num_iterations;

	// Initialize elapsed time vars
	double aligned_elapsed_time = 0.0;
	double total_aligned_elapsed_time = 0.0;
	double avg_aligned_elapsed_time = 0.0;
	double aligned_standard_dev = 0.0;
	double *aligned_omp_run_timings;

	// Set up other time vars
        struct timespec start_time;
        struct timespec end_time;

	// Initialize matrices
	double **omp_aligned_matrix_A = (double**)aligned_alloc(alignment, m_row);
        double **omp_aligned_matrix_B = (double**)aligned_alloc(alignment, n_row);
	double **omp_aligned_matrix_C = (double**)aligned_alloc(alignment, m_row);
	for (i_omp_aligned=0; i_omp_aligned<m_omp_aligned; i_omp_aligned++){
            omp_aligned_matrix_A[i_omp_aligned] = (double*)aligned_alloc(alignment, n_col);
            omp_aligned_matrix_C[i_omp_aligned] = (double*)aligned_alloc(alignment, k_col);
	}
	for (i_omp_aligned=0; i_omp_aligned<n_omp_aligned; i_omp_aligned++){
            omp_aligned_matrix_B[i_omp_aligned] = (double*)aligned_alloc(alignment, k_col);
	}

	// Setup array to hold all timing data
	aligned_omp_run_timings = (double*)malloc(num_aligned_iterations * sizeof(double));

	// Populate matrices A and B with data
        ompPopulateMatrix(omp_aligned_matrix_A, &m_omp_aligned, &n_omp_aligned, seed_omp_aligned);
        ompPopulateMatrix(omp_aligned_matrix_B, &n_omp_aligned, &k_omp_aligned, seed_omp_aligned);

	// Matrix multiply
        #pragma omp for
	for (i=0; i<num_aligned_iterations; i++){
            clock_gettime(CLOCK_REALTIME, &start_time);
            ompSIMDMatrixMultiply(omp_aligned_matrix_A, omp_aligned_matrix_B, omp_aligned_matrix_C, m_omp_aligned, n_omp_aligned, k_omp_aligned);
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
	    resetMatrix(omp_aligned_matrix_C, m_omp_aligned, k_omp_aligned);
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
