/*
function to partition into n section according to the modulo of an array element
*/
void generalized_modulo_partition(size_t *idx_begin, size_t *idx_end, size_t const *begin, int *offset, int n_segment, int multiplier = 1) {
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
		bool parallel = num_threads < omp_get_num_threads() / 2;
		#pragma omp parallel for schedule(static) num_threads(num_threads)
		for (int i = 0; i < n_partition; ++i) {
			
			/* compute limits */
			size_t lower = (n_segment * i) / n_partition; 
			size_t middle = (n_segment * (2*i + 1)) / (n_partition * 2); 
			size_t upper = (n_segment * (i + 1)) / n_partition; 

			/* actually partition */
			size_t *partitioned_it;
			if (lower < middle && middle < upper) {
				if (parallel) {
					partitioned_it  = __gnu_parallel::partition(idx_begin + offset[lower], idx_begin + offset[upper],
					[&](size_t const idx){
						return (begin[idx] / multiplier) % n_segment < middle;
					});
				} else
					partitioned_it = std::partition(idx_begin + offset[lower], idx_begin + offset[upper],
					[&](size_t const idx){
						return (begin[idx] / multiplier) % n_segment < middle;
					});
				offset[middle] = std::distance(idx_begin, partitioned_it);
			}
		}
	}
}