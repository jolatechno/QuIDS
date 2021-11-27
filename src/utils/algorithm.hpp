#include <vector>

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
template <class idType, class idIteratorType, class countIteratorType, class functionType>
void generalized_partition(idType const id_begin, idType const id_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
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

	std::vector<size_t> count(n_segment*num_threads, 0);

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = partitioner(i);
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	#pragma omp parallel for schedule(static)
	for (int i = 0; i < n_segment; ++i)
		offset[i + 1] = count[i*num_threads + num_threads - 1];

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = partitioner(i);
			idx_out[--count[key*num_threads + thread_id]] = i;
		}
	}
}