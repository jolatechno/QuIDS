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
template <class idType, class valueType, class countType>
void generalized_modulo_partition_power_of_two(idType const id_begin, idType const id_end, idType *idx_out, valueType const *begin, countType *offset, int const n_segment) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end - id_begin;

	if (n_segment == 1)
		return;

	const valueType bitmask = n_segment - 1;

	/* 
	number of threads
	*/
	const int num_threads = []() {
		/* get num thread */
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		return num_threads;
	}();

	countType *count = new countType[n_segment*num_threads]();

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

	delete[] count;
}

template <class idType, class valueType, class countType>
void generalized_modulo_partition(idType const id_begin, idType const id_end, idType *idx_out, valueType const *begin, countType *offset, int const n_segment) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end - id_begin;

	if (n_segment == 1)
		return;

	/* 
	number of threads
	*/
	const int num_threads = []() {
		/* get num thread */
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		return num_threads;
	}();

	countType *count = new countType[n_segment*num_threads]();

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = begin[i] % n_segment;
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
			auto key = begin[i] % n_segment;
			idx_out[--count[key*num_threads + thread_id]] = i;
		}
	}

	delete[] count;
}
