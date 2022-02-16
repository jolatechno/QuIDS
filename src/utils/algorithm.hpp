#pragma once

#include <vector>

#ifndef NTH_ELEMENT_SEGMENT_RATIO
	#define NTH_ELEMENT_SEGMENT_RATIO 100
#endif
#ifndef UPSIZE_POLICY
	#define UPSIZE_POLICY 1.1
#endif
#ifndef DOWNSIZE_POLICY
	#define DOWNSIZE_POLICY 0.85
#endif
#ifndef MIN_VECTOR_SIZE
	#define MIN_VECTOR_SIZE 1000
#endif

// global variable definition
int nth_element_segment_ratio = NTH_ELEMENT_SEGMENT_RATIO;
float upsize_policy = UPSIZE_POLICY;
float downsize_policy = DOWNSIZE_POLICY;
size_t min_vector_size = MIN_VECTOR_SIZE;

/*
smarter resize
*/
template<class VectorType>
void smart_resize(VectorType &vector, size_t size) {
	if (size < min_vector_size)
		size = min_vector_size;

	if (size <= vector.size()) {
		vector.resize(size);

		if (size*upsize_policy < vector.capacity()*downsize_policy)
			vector.reserve(size);
	} else {
		vector.reserve(size*upsize_policy);
		vector.resize(size);
	}
}

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
approximate nth element
*/
template <class idIteratorType, class functionType>
void aprox_nth_element(idIteratorType begin, idIteratorType middle, idIteratorType end, functionType comparator) {
	size_t const n_select = std::distance(begin, middle);
	size_t const size = std::distance(begin, end);

	size_t const segment_size = std::min(size, (size*nth_element_segment_ratio) / n_select);
	size_t const n_segment = size / segment_size;
	size_t const per_segment_select = n_select / n_segment;

	for (size_t i = 0; i < n_segment; ++i) {
		size_t this_begin = segment_size * i;
		size_t this_end = segment_size * (i + 1);

		std::nth_element(begin + this_begin, begin + this_begin + per_segment_select, begin + this_end, comparator);
	}

	for (size_t i = 1; i < n_segment; ++i) {
		size_t segment_begin = segment_size * i;
		size_t segment_end = segment_size * (i + 1);
		size_t segment_destination = per_segment_select * i;

		std::copy(begin + segment_begin, begin + segment_begin, begin + segment_destination);
	}
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
parallel approximate nth element
*/
template <class idIteratorType, class functionType>
void parallel_aprox_nth_element(idIteratorType begin, idIteratorType middle, idIteratorType end, functionType comparator) {
	size_t const n_select = std::distance(begin, middle);
	size_t const size = std::distance(begin, end);

	/* number of threads */
	int num_threads;
	#pragma omp parallel
	#pragma omp single
	num_threads = omp_get_num_threads();

	size_t const segment_size = std::min(size / num_threads, (size*nth_element_segment_ratio) / n_select);
	size_t const n_segment = size / segment_size;
	size_t const per_segment_select = n_select / n_segment;

	#pragma omp parallel for
	for (size_t i = 0; i < n_segment; ++i) {
		size_t this_begin = segment_size * i;
		size_t this_end = segment_size * (i + 1);

		std::nth_element(begin + this_begin, begin + this_begin + per_segment_select, begin + this_end, comparator);
	}

	for (size_t i = 1; i < n_segment; ++i) {
		size_t segment_begin = segment_size * i;
		size_t segment_end = segment_size * (i + 1);
		size_t segment_destination = per_segment_select * i;

		std::copy(begin + segment_begin, begin + segment_begin, begin + segment_destination);
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