#include <vector>

/*
closest power of two
*/
int nearest_power_of_two(int n) {
	for (int i = 1;; i *= 2)
		if (i >= n)
			return i;
}

int log_2_upper_bound(int n) {
	for (int i = 1;; ++i)
		if (n >> i == 0)
			return i;
}

/* 
parallel iota
*/

template <class iteratorType, class valueType>
void parallel_iota(iteratorType begin, iteratorType end, const valueType value_begin) {
	size_t distance = std::distance(begin, end);

	if (value_begin == 0) {
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < distance; ++i)
			begin[i] = i;
	} else
		#pragma omp parallel for schedule(static)
		for (size_t i = 0; i < distance; ++i)
			begin[i] = value_begin + i;
}

/*
function to partition into n section
*/
template <class idIteratorType, class countIteratorType, class functionType>
void generalized_partition(const idIteratorType idx_in, const idIteratorType idx_in_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
	auto id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end;

	if (n_segment == 1) {
		std::copy(idx_in, idx_in_end, idx_out);
		return;
	}

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
			auto key = partitioner(idx_in[i]);
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (auto i = 0; i < id_end; ++i) {
			auto idx = idx_in[i];
			auto key = partitioner(idx);
			idx_out[--count[key*num_threads + thread_id]] = idx;
		}
	}

	#pragma omp parallel for schedule(static)
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];
}

/*
function to partition into n section
*/
template <class idType, class idIteratorType, class countIteratorType, class functionType>
void generalized_partition_from_iota(const idType id_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end;

	if (n_segment == 1) {
		std::iota(idx_out, idx_out + id_end, 0);
		return;
	}

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

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (auto i = 0; i < id_end; ++i) {
			auto key = partitioner(i);
			idx_out[--count[key*num_threads + thread_id]] = i;
		}
	}

	#pragma omp parallel for schedule(static)
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];
}

/*
function to partition into n section
*/
template <class idIteratorType, class countIteratorType, class functionType>
void single_threaded_generalized_partition(const idIteratorType idx_in, const idIteratorType idx_in_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
	auto id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end;

	if (n_segment == 1) {
		std::copy(idx_in, idx_in_end, idx_out);
		return;
	}

	std::fill(offset, offset + n_segment, 0);

	for (auto i = 0; i < id_end; ++i) {
		auto key = partitioner(idx_in[i]);
		++offset[key];
	}
		
	std::partial_sum(offset, offset + n_segment, offset);

	for (auto i = 0; i < id_end; ++i) {
		auto idx = idx_in[i];
		auto key = partitioner(idx);
		idx_out[--offset[key]] = idx;
	}
}

/*
function to complete an aldready partial partition into n section
*/
template <class idIteratorType, class countIteratorType, class functionType>
void complete_generalized_partition(const idIteratorType idx_in, const idIteratorType idx_in_end, idIteratorType idx_out, countIteratorType offset, int const n_segment, functionType const partitioner) {
	auto id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	offset[0] = 0;
	
	if (n_segment == 1) {
		std::copy(idx_in, idx_in_end, idx_out);
		offset[n_segment] = id_end;
		return;
	}

	/* number of threads */
	int num_threads;
	#pragma omp parallel
	#pragma omp single
	num_threads = omp_get_num_threads();

	/* get copy of the first indexes */
	auto id_begin = offset[n_segment];

	std::vector<size_t> count(n_segment*num_threads, 0);

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto key = partitioner(idx_in[i]);
			++count[key*num_threads + thread_id];
		}

		/* add initial count to this count */
		#pragma omp for schedule(static)
		for (int i = 0; i < n_segment; ++i)
			count[(i + 1)*num_threads] += offset[i + 1] - offset[i];
	}

	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	/* copy old indexes into the gaps */
	for (int i = 0; i < n_segment; ++i) {
		long long int begin = offset[i];
		long long int end = offset[i + 1];
		long long int j_offset = count[i*num_threads + num_threads - 1] - begin;
		
		#pragma omp parallel for schedule(static)
		for (long long int j = begin; j < end; ++j)
			idx_out[j + j_offset] = idx_in[j];
	}

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (auto i = id_begin; i < id_end; ++i) {
			auto idx = idx_in[i];
			auto key = partitioner(idx);
			idx_out[--count[key*num_threads + thread_id]] = idx;
		}
	}

	#pragma omp parallel for schedule(static)
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];
	offset[n_segment] = id_end;
}