#pragma once

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
			return i - 1;
}

/* 
parallel iota
*/
template <class iteratorType, class valueType>
void parallel_iota(iteratorType begin, iteratorType end, const valueType value_begin) {
	size_t distance = std::distance(begin, end);

	if (value_begin == 0) {
		#pragma omp parallel for 
		for (size_t i = 0; i < distance; ++i)
			begin[i] = i;
	} else
		#pragma omp parallel for 
		for (size_t i = 0; i < distance; ++i)
			begin[i] = value_begin + i;
}

/*
function to partition into n section
*/
template <class idIteratorType, class countIteratorType, class functionType>
void generalized_partition(idIteratorType idx_in, idIteratorType idx_in_end, idIteratorType idx_buffer,
	countIteratorType offset, countIteratorType offset_end,
	functionType const partitioner) {
	
	int const n_segment = std::distance(offset, offset_end) - 1;
	long long int const id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	std::fill(offset, offset_end - 1, 0);
	offset[n_segment] = id_end;

	if (n_segment == 1)
		return;
	if (id_end == 0)
		return;

	for (long long int i = id_end - 1; i >= 0; --i) {
		auto key = partitioner(idx_in[i]);
		++offset[key];
	}
	
	std::partial_sum(offset, offset + n_segment, offset);

	for (long long int i = id_end - 1; i >= 0; --i) {
		auto idx = idx_in[i];
		auto key = partitioner(idx);
		idx_buffer[--offset[key]] = idx;
	}

	std::copy(idx_buffer, idx_buffer + id_end, idx_in);
}

/*
function to partition into n section
*/
template <class idIteratorType, class idType, class countIteratorType, class functionType>
void generalized_partition_from_iota(idIteratorType idx_in, idIteratorType idx_in_end, idType const iotaOffset,
	countIteratorType offset, countIteratorType offset_end,
	functionType const partitioner) {

	long long int iota_offset = iotaOffset;
	int const n_segment = std::distance(offset, offset_end) - 1;
	long long int const id_end = std::distance(idx_in, idx_in_end);

	/* limit values */
	std::fill(offset, offset_end - 1, 0);
	offset[n_segment] = id_end;

	if (n_segment == 1) {
		parallel_iota(idx_in, idx_in_end, iota_offset);
		return;
	}
	if (id_end == 0)
		return;

	for (long long int i = id_end + iota_offset - 1; i >= iota_offset; --i) {
		auto key = partitioner(i);
		++offset[key];
	}	
	
	std::partial_sum(offset, offset + n_segment, offset);

	for (long long int i = id_end + iota_offset - 1; i >= iota_offset; --i) {
		auto key = partitioner(i);
		idx_in[--offset[key]] = i;
	}
}


/*
function to partition into n section
*/
template <class idIteratorType, class countIteratorType, class functionType>
void parallel_generalized_partition(idIteratorType idx_in, idIteratorType idx_in_end, idIteratorType idx_buffer,
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

		#pragma omp for 
		for (long long int i = id_end - 1; i >= 0; --i) {
			auto key = partitioner(idx_in[i]);
			++count[key*num_threads + thread_id];
		}
	}
	
	__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment*num_threads, count.begin());

	#pragma omp parallel
	{
		int thread_id = omp_get_thread_num();
		
		#pragma omp for 
		for (long long int i = id_end - 1; i >= 0; --i) {
			auto idx = idx_in[i];
			auto key = partitioner(idx);
			idx_buffer[--count[key*num_threads + thread_id]] = idx;
		}
	}

	#pragma omp parallel for 
	for (int i = 1; i < n_segment; ++i)
		offset[i] = count[i*num_threads];

	std::copy(idx_buffer, idx_buffer + id_end, idx_in);
}