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
function to partition into n section
*/
template <class idType, class idIteratorType, class countIteratorType, class functionType>
void generalized_partition(idType const id_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end;

	if (n_segment == 1)
		return;

	/* number of threads */
	int num_threads;
	#pragma omp parallel
	#pragma omp single
	num_threads = omp_get_num_threads();

	std::vector<size_t> count(n_segment*num_threads, 0);

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = 0; i < id_end; ++i) {
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
		for (auto i = 0; i < id_end; ++i) {
			auto key = partitioner(i);
			idx_out[--count[key*num_threads + thread_id]] = i;
		}
	}
}

/*
function to complete an aldready partial partition into n section
*/
template <class idType, class idIteratorType, class countIteratorType, class functionType>
void complete_generalized_partition(idType const id_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
	/* limit values */
	offset[0] = 0;
	
	if (n_segment == 1) {
		offset[n_segment] = id_end;
		return;
	}

	/* number of threads */
	int num_threads;
	#pragma omp parallel
	#pragma omp single
	num_threads = omp_get_num_threads();

	/* get old count */

	std::vector<size_t> count(n_segment*num_threads, 0);
	size_t id_begin = offset[n_segment];

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = partitioner(i);
			++count[key*num_threads + thread_id];
		}

		/* add initial count to this count */
		#pragma omp for schedule(static)
		for (int i = 0; i < n_segment; ++i)
			count[i*num_threads + num_threads - 1] += offset[i + 1] - offset[i];
	}

	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	/* get copy of the first indexes */
	std::vector<size_t> old_idx(id_begin, 0);
	#pragma omp parallel for schedule(static)
	for (auto i = 0; i < id_begin; ++i)
		old_idx[i] = idx_out[i];

	/* copy old indexes into the gaps */
	for (int i = n_segment - 1; i >= 0; --i) {
		long long int begin = offset[i];
		long long int end = offset[i + 1];

		offset[i + 1] = count[i*num_threads + num_threads - 1];

		long long int j_offset = count[i*num_threads + num_threads - 1] - end;
		count[i*num_threads + num_threads - 1] -= end - begin;

		#pragma omp parallel for schedule(static)
		for (long long int j = end - 1; j >= begin; --j)
			idx_out[j + j_offset] = old_idx[j];
	}

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