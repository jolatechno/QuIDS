#include <mpi.h>

MPI_Status global_status;

void generalized_modulo_partition(size_t *idx_begin, size_t *idx_end, size_t const *begin, int *offset, int n_segment) {
	/* should realy be optimized */
	offset[0] = 0;
	offset[n_segment] = std::distance(idx_begin, idx_end);

	for (int i = 0; i < n_segment - 1; ++i) {
		auto partitioned_it = __gnu_parallel::partition(idx_begin + offset[i], idx_end,
			[&](size_t const idx){
				return begin[idx] % n_segment == i;
			});
		offset[i + 1] = std::distance(idx_begin, partitioned_it);
	}
}

int clossest_power_of_two(int x) {
	int y = 1;
	for (; y < x; y *= 2) {}
	return y;
}
