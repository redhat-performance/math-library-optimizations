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

#define MAX_MAT_VALUE 100000
#define ALIGNMENT 16
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
                mat_C[a][c] = (mat_C[a][c]) + (mat_A[a][c] * mat_B[a][c]);
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
                mat_C[a][c] = (mat_C[a][c]) + (mat_A[a][c] * mat_B[a][c]);
	    }
	}
    }
};


int main(int argc, char *argv[]){

    // Initialize vars
    int m, n, k, i, j, seed;

    // Make sure the user passed in 'm', 'n', and 'k'
    if (argc < 4){
        printf("\nIncomplete set of arguments. Required arguments: 'm', 'n', and 'k'.\n");
        return 1;
    }
    else if (argc > 5){
        printf("\nToo many arguments. Required arguments: 'm', 'n', and 'k'. Optional argument: 'seed'\n");
        return 1;
    }

    // User inputs succeeded. Now parse input
    char *p;
    m = strtol(argv[1], &p, 10);
    n = strtol(argv[2], &p, 10);
    k = strtol(argv[3], &p, 10);

    // If the user passed in a seed, let's check it
    if (argc == 5)
        seed = strtol(argv[4], &p, 10);
    else
        seed = DEFAULT_SEED;

    // Print matrix info (MxN matrix)
    printf("Matrix info:\n");
    printf("  Matrix A:\n");
    printf("      M=%d\n", m);
    printf("      N=%d\n", n);
    printf("  Matrix B:\n");
    printf("      N=%d\n", n);
    printf("      K=%d\n\n", k);

    // Print seed info
    printf("Seed info:\n");
    if (argc == 4)
        printf("  Using predefined seed = DEFAULT_SEED = %d\n", seed);
    else
        printf("  Using seed = %d\n", seed);

    // Set up time vars
    struct timespec t1_start, t1_end, t2_start, t2_end, t3_start, t3_end;
    double elapsed_time;

    // Setup matrices
    double **omp_nonaligned_matrix_A;
    double **omp_nonaligned_matrix_B;
    double **omp_nonaligned_matrix_C;

    clock_gettime(CLOCK_REALTIME, &t1_start);

    int m_omp_nonaligned, n_omp_nonaligned, k_omp_nonaligned, seed_omp_nonaligned;

    #pragma omp parallel shared(omp_nonaligned_matrix_A, omp_nonaligned_matrix_B, omp_nonaligned_matrix_C) private(i, m_omp_nonaligned, n_omp_nonaligned, k_omp_nonaligned, seed_omp_nonaligned)
    {

        // Set values for m, n, and k
        m_omp_nonaligned = m;
	n_omp_nonaligned = n;
	k_omp_nonaligned = k;

        // Do the same with 'seed'
        seed_omp_nonaligned = seed;

	// Do not use dynamic threading
        omp_set_dynamic(0);

	omp_nonaligned_matrix_A = (double **)malloc(m * sizeof(double*));
	omp_nonaligned_matrix_B = (double **)malloc(n * sizeof(double*));
	omp_nonaligned_matrix_C = (double **)malloc(m * sizeof(double*));
	for (i=0; i<m; i++)
            omp_nonaligned_matrix_A[i] = (double *)malloc(n * sizeof(double));

	for (i=0; i<n; i++)
            omp_nonaligned_matrix_B[i] = (double *)malloc(k * sizeof(double));

	for (i=0; i<m; i++)
            omp_nonaligned_matrix_C[i] = (double *)malloc(k * sizeof(double));

	// Populate matrices A and B with data
        ompPopulateMatrix(omp_nonaligned_matrix_A, &m_omp_nonaligned, &n_omp_nonaligned, seed_omp_nonaligned);
        ompPopulateMatrix(omp_nonaligned_matrix_B, &n_omp_nonaligned, &k_omp_nonaligned, seed_omp_nonaligned);

	// Matrix multiply
        ompMatrixMultiply(omp_nonaligned_matrix_A, omp_nonaligned_matrix_B, omp_nonaligned_matrix_C, m_omp_nonaligned, n_omp_nonaligned, k_omp_nonaligned);
    }	

    clock_gettime(CLOCK_REALTIME, &t1_end);
    elapsed_time = (t1_end.tv_sec - t1_start.tv_sec) * 1000.0;
    elapsed_time += (t1_end.tv_nsec - t1_start.tv_nsec) / 1000000.0;
    elapsed_time /= 1000.0;
    printf("Non-Aligned OpenMP took %0.3f sec\n", elapsed_time);

    // Free matrices
    for (i=0; i<n; i++) {
        free(omp_nonaligned_matrix_B[i]);
        free(omp_nonaligned_matrix_C[i]);
    }
    for (i=0; i<m; i++)
        free(omp_nonaligned_matrix_A[i]);

    // Initialize vars
    int i_omp_aligned;
    int m_omp_aligned;
    int n_omp_aligned;
    int k_omp_aligned;
    int seed_omp_aligned;
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

    clock_gettime(CLOCK_REALTIME, &t2_start);

    #pragma omp parallel shared(m_row, n_col, n_row, k_col, alignment) private(i_omp_aligned)
    {
	// Do not use dynamic threading
        omp_set_dynamic(0);

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

	// Populate matrices A and B with data
        ompPopulateMatrix(omp_aligned_matrix_A, &m_omp_aligned, &n_omp_aligned, seed_omp_aligned);
        ompPopulateMatrix(omp_aligned_matrix_B, &n_omp_aligned, &k_omp_aligned, seed_omp_aligned);

	// Matrix multiply
        ompSIMDMatrixMultiply(omp_aligned_matrix_A, omp_aligned_matrix_B, omp_aligned_matrix_C, m_omp_aligned, n_omp_aligned, k_omp_aligned);
    }
	
    clock_gettime(CLOCK_REALTIME, &t2_end);
    elapsed_time = (t2_end.tv_sec - t2_start.tv_sec) * 1000.0;
    elapsed_time += (t2_end.tv_nsec - t2_start.tv_nsec) / 1000000.0;
    elapsed_time /= 1000.0;
    printf("Aligned OpenMP took %0.3f sec\n", elapsed_time);

    // Initialize regular matrices
    double **matrix_A;
    double **matrix_B;
    double **matrix_C;

    clock_gettime(CLOCK_REALTIME, &t3_start);
    matrix_A = (double **)malloc(m * sizeof(double*));
    matrix_B = (double **)malloc(n * sizeof(double*));
    matrix_C = (double **)malloc(m * sizeof(double*));
    for (i=0; i<m; i++)
        matrix_A[i] = (double *)malloc(n * sizeof(double));
        
    for (i=0; i<n; i++)
        matrix_B[i] = (double *)malloc(k * sizeof(double));
        
    for (i=0; i<m; i++)
        matrix_C[i] = (double *)malloc(k * sizeof(double));

    // Populate the matrices with random values
    populateMatrix(matrix_A, m, n, seed);
    populateMatrix(matrix_B, n, k, seed);

    // Multiply matrix_A and matrix_B to get matrix_C
    matrixMultiply(matrix_A, matrix_B, matrix_C, m, n, k);

    clock_gettime(CLOCK_REALTIME, &t3_end);
    elapsed_time = (t3_end.tv_sec - t3_start.tv_sec) * 1000.0;
    elapsed_time += (t3_end.tv_nsec - t3_start.tv_nsec) / 1000000.0;
    elapsed_time /= 1000.0;
    printf("Non-OpenMP took %0.3f sec\n", elapsed_time);

    // Free matrices
    for (i=0; i<n; i++) {
        free(matrix_B[i]);
        free(matrix_C[i]);
    }
    for (i=0; i<m; i++)
        free(matrix_A[i]);

    return 0;
}
