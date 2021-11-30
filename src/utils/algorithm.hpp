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
void stable_generalized_partition(idIteratorType idx_in, idIteratorType idx_in_end, idIteratorType idx_buffer,
	countIteratorType offset, countIteratorType offset_end,
	functionType const partitioner) {
	
	int const n_segment = std::distance(offset, offset_end) - 1;
	long long int const id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end;

	if (n_segment == 1)
		return;
	if (id_end == 0) {
		std::fill(offset, offset_end, 0);
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
		for (long long int i = id_end - 1; i >= 0; --i) {
			auto key = partitioner(idx_in[i]);
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (long long int i = id_end - 1; i >= 0; --i) {
			auto idx = idx_in[i];
			auto key = partitioner(idx);
			idx_buffer[--count[key*num_threads + thread_id]] = idx;
		}
	}

	#pragma omp parallel for schedule(static)
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];

	std::copy(idx_buffer, idx_buffer + id_end, idx_in);
}

/*
function to complete an aldready partial partition into n section
*/
template <class idIteratorType, class countIteratorType, class functionType>
void complete_stable_generalized_partition(idIteratorType idx_in, idIteratorType idx_in_end, idIteratorType idx_buffer,
	countIteratorType offset, countIteratorType offset_end,
	functionType const partitioner) {

	int const n_segment = std::distance(offset, offset_end) - 1;
	long long int const id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	offset[0] = 0;
	
	if (n_segment == 1) {
		std::copy(idx_in, idx_in_end, idx_buffer);
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

	if (id_end == id_begin)
		return;
	if (id_begin == 0)
		return stable_generalized_partition(idx_in, idx_in_end, idx_buffer, offset, offset_end, partitioner);

	std::vector<size_t> count(n_segment*num_threads, 0);

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();

		#pragma omp for schedule(static)
		for (long long int i = id_end - 1; i >= id_begin; --i) {
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
			idx_buffer[j + j_offset] = idx_in[j];
	}

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (long long int i = id_end - 1; i >= id_begin; --i) {
			auto idx = idx_in[i];
			auto key = partitioner(idx);
			idx_buffer[--count[key*num_threads + thread_id]] = idx;
		}
	}

	#pragma omp parallel for schedule(static)
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];
	offset[n_segment] = id_end;

	std::copy(idx_buffer, idx_buffer + id_end, idx_in);
}

/*
function to partition into n section
*/
template <class idIteratorType, class idType, class countIteratorType, class functionType>
void stable_generalized_partition_from_iota(idIteratorType idx_in, idIteratorType idx_in_end, idType const offset_iota,
	countIteratorType offset, countIteratorType offset_end,
	functionType const partitioner) {
	
	int const n_segment = std::distance(offset, offset_end) - 1;
	long long int const id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	offset[0] = 0;
	offset[n_segment] = id_end;

	if (n_segment == 1) {
		parallel_iota(idx_in, idx_in_end, 0);
		return;
	}
	if (id_end == 0) {
		std::fill(offset, offset_end, 0);
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
		for (long long int i = id_end + offset_iota - 1; i >= offset_iota; --i) {
			auto key = partitioner(i);
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for schedule(static)
		for (long long int i = id_end + offset_iota - 1; i >= offset_iota; --i) {
			auto key = partitioner(i);
			idx_in[--count[key*num_threads + thread_id]] = i;
		}
	}

	#pragma omp parallel for schedule(static)
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];
}

/*
function to partition according to one partioner while presaving another partitioner
*/
template <class idIteratorType, class countIteratorType, class functionType, class functionType2>
idIteratorType partition_conserve_stable_partition(idIteratorType idx_in, idIteratorType idx_in_end, idIteratorType idx_buffer,
	countIteratorType offset, countIteratorType offset_end,
	functionType const partitioner, functionType2 const old_partitioner) {

	idIteratorType partitioned_it = std::stable_partition(idx_in, idx_in_end, partitioner);

	size_t first_segment = std::distance(idx_in, partitioned_it);
	size_t second_segment = std::distance(partitioned_it, idx_in_end);

	if (first_segment > second_segment) {
		const int n_segment = std::distance(offset, offset_end) - 1;

		std::vector<size_t> partial_offset(n_segment + 1, 0);
		stable_generalized_partition(partitioned_it, idx_in_end, idx_buffer,
			partial_offset.begin(), partial_offset.begin() + n_segment + 1,
			old_partitioner);

		#pragma omp parallel for schedule(static)
		for (int i = 1; i <= n_segment; ++i)
			offset[i] -= partial_offset[i];
	} else {
		stable_generalized_partition(idx_in, partitioned_it, idx_buffer,
			offset, offset_end,
			old_partitioner);
	}

	return partitioned_it;
}