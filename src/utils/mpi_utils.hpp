#include <mpi.h>

MPI_Status global_status;

/*
function to partition into n section according to the modulo of an array element
*/
void generalized_modulo_partition(size_t *idx_begin, size_t *idx_end, size_t const *begin, int *offset, int n_segment) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = std::distance(idx_begin, idx_end);

	/* control omp nested parallelism */
	omp_set_dynamic(0);
	omp_set_nested(true);

	/* recursivly partition */
	for (int n_partition = 1; n_partition < n_segment; n_partition *= 2) {

		/* nested for loop */
		int num_threads = std::min(n_partition, omp_get_num_threads());
		#pragma omp parallel for num_threads(num_threads)
		for (int i = 0; i < n_partition; ++i) {
			
			/* compute limits */
			size_t lower = (n_segment * i) / n_partition; 
			size_t middle = (n_segment * (2*i + 1)) / (n_partition * 2); 
			size_t upper = (n_segment * (i + 1)) / n_partition; 

			/* actually partition */
			if (lower < middle && middle < upper) {
				auto partitioned_it = __gnu_parallel::partition(idx_begin + offset[lower], idx_begin + offset[upper],
				[&](size_t const idx){
					return begin[idx] % n_segment < middle;
				});
				offset[middle] = std::distance(idx_begin, partitioned_it);
			}
		}
	}
}

/*
function to partition into pair of almost equal sum
*/
void make_equal_pairs(size_t *size_begin, size_t *size_end, int *pair_id) {
	size_t size = std::distance(size_begin, size_end);

	int *node_ids = new int[size];
	std::iota(node_ids, node_ids + size, 0);

	/* compute average value */
	__gnu_parallel::sort(node_ids, node_ids + size,
		[&](int const node_id1, int const node_id2) {
			return size_begin[node_id1] > size_begin[node_id2];
		});

	#pragma omp parallel for
	for (int i = 0; i < size; ++i)
		pair_id[node_ids[i]] = pair_id[node_ids[size - i - 1]];
} 

/*
get mpi type
*/
MPI_Datatype get_mpi_datatype(float x) { return MPI::FLOAT; }
MPI_Datatype get_mpi_datatype(double x) { return MPI::DOUBLE; }
MPI_Datatype get_mpi_datatype(long double x) { return MPI::LONG_DOUBLE; }
MPI_Datatype get_mpi_datatype(std::complex<float> x) { return MPI::COMPLEX; }
MPI_Datatype get_mpi_datatype(std::complex<double> x) { return MPI::DOUBLE_COMPLEX; }
MPI_Datatype get_mpi_datatype(std::complex<long double> x) { return MPI::LONG_DOUBLE_COMPLEX; }
MPI_Datatype get_mpi_datatype(bool x) { return MPI::BOOL; }
MPI_Datatype get_mpi_datatype(char x) { return MPI::CHAR; }
MPI_Datatype get_mpi_datatype(signed char x) { return MPI::SIGNED_CHAR; }
MPI_Datatype get_mpi_datatype(unsigned char x) { return MPI::UNSIGNED_CHAR; }
MPI_Datatype get_mpi_datatype(short x) { return MPI::SHORT; }
MPI_Datatype get_mpi_datatype(unsigned short x) { return MPI::UNSIGNED_SHORT; }
MPI_Datatype get_mpi_datatype(int x) { return MPI::INT; }
MPI_Datatype get_mpi_datatype(unsigned int x) { return MPI::UNSIGNED; }
MPI_Datatype get_mpi_datatype(long x) { return MPI::LONG; }
MPI_Datatype get_mpi_datatype(unsigned long x) { return MPI::UNSIGNED_LONG; }