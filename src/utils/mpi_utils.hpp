#include <mpi.h>

MPI_Status global_status;

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

int make_equal_pairs(size_t *size_begin, size_t *size_end, int *pair_id) {
	size_t size = std::distance(size_begin, size_end);

	/* compute average value */
	long long int avg_size = __gnu_parallel::accumulate(size_begin, size_end, 0) / size;

	/* if size is odd, find the node to not pair */
	int alone_node = size;
	if (size % 2 == 1) {
		auto min_it = __gnu_parallel::min_element(size_begin + size / 2, size_end,
			[&](size_t size1, size_t size2) {
				long long int diff1 = std::abs((long long int)avg_size - (long long int)size1);
				long long int diff2 = std::abs((long long int)avg_size - (long long int)size2);
				return diff1 < diff2;
			});
		alone_node = std::distance(size_begin, min_it);
	}

	/* initial guess */
	#pragma omp parallel for
	for (int i = 0; i < size / 2; ++i) {
		pair_id[i] = size / 2 + i;
		pair_id[i] += pair_id[i] >= alone_node;
	}

	/* iterativly improve the guess */
	if (size / 2 > 1)
		for (size_t step = 0; step < 10*size*size; ++step) {
			/* generate two random indexes */
			int i = rand() % (size / 2);
			int j = rand() % (size / 2 - 1);
			j += i == j;

			/* compute starting average values */
			long long int size_diff_i = std::abs(avg_size - (long long int)(size_begin[i] + size_begin[pair_id[i]]) / 2);
			long long int size_diff_j = std::abs(avg_size - (long long int)(size_begin[j] + size_begin[pair_id[j]]) / 2);

			/* compute average values after swap */
			long long int size_diff_i_swaped = std::abs(avg_size - (long long int)(size_begin[i] + size_begin[pair_id[j]]) / 2);
			long long int size_diff_j_swaped = std::abs(avg_size - (long long int)(size_begin[j] + size_begin[pair_id[i]]) / 2); 

			/* swap if necessary */
			if (std::max(size_diff_i_swaped, size_diff_j_swaped) < std::max(size_diff_i, size_diff_j))
				std::swap(pair_id[i], pair_id[j]);
		}

	return alone_node == size ? -1 : alone_node;
} 