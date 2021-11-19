/*
closest power of two
*/
int nearest_power_of_two(int n) {
	for (int i = 1;; i *= 2)
		if (i >= n)
			return i;
}

int modulo_2_upper_bound(int n) {
	for (int i = 1;; ++i)
		if (n >> i == 0)
			return i;
}

/*
function to partition into n section according to the modulo of an array element

!!!! n_segment MUST be a power of two !!!!
*/
void generalized_modulo_partition_power_of_two(size_t const id_begin, size_t const id_end, size_t *idx_out, size_t const *begin, int *offset, int const n_segment) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end - id_begin;

	if (n_segment == 1)
		return;

	const size_t bitmask = n_segment - 1;

	/* 
	number of threads
	*/
	const size_t num_threads = []() {
		/* get num thread */
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		return num_threads;
	}();

	int *count = new int[n_segment*num_threads]();

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = begin[i] & bitmask;
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count, count + n_segment*num_threads, count);

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_segment; ++i)
		offset[i + 1] = count[i*num_threads + num_threads - 1];

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = begin[i] & bitmask;
			idx_out[--count[key*num_threads + thread_id]] = i;
		}
	}
}

/*
function to partition into n section according to the modulo of an array element

!!!! n_segment MUST be a power of two !!!!
*/
void generalized_shifted_modulo_partition_power_of_two(size_t const id_begin, size_t const id_end, size_t *idx_out, size_t const *begin, int *offset, int const n_segment, int const shift) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end - id_begin;

	if (n_segment == 1)
		return;

	const size_t bitmask = n_segment - 1;

	/* 
	number of threads
	*/
	const size_t num_threads = []() {
		/* get num thread */
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		return num_threads;
	}();

	int *count = new int[n_segment*num_threads]();

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = (begin[i] << shift) & bitmask;
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count, count + n_segment*num_threads, count);

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_segment; ++i)
		offset[i + 1] = count[i*num_threads + num_threads - 1];

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = (begin[i] << shift) & bitmask;
			idx_out[--count[key*num_threads + thread_id]] = i;
		}
	}
}
