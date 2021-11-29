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

	if (n_segment <= num_threads) {
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
	} else {
		std::fill(offset, offset + n_segment, 0);

		for (auto i = 0; i < id_end; ++i) {
			auto key = partitioner(i);
			++offset[key];
		}
		
		__gnu_parallel::partial_sum(offset, offset + n_segment, offset);

		for (auto i = 0; i < id_end; ++i) {
			auto key = partitioner(i);
			idx_out[--offset[key]] = i;
		}

		offset[n_segment] = id_end;
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

	/* get copy of the first indexes */
	auto id_begin = offset[n_segment];
	std::vector<size_t> old_idx(id_begin, 0);
	#pragma omp parallel for schedule(static)
	for (auto i = 0; i < id_begin; ++i)
		old_idx[i] = idx_out[i];

	if (n_segment <= num_threads) {
		std::vector<size_t> count(n_segment*num_threads, 0);

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

		#pragma omp parallel for schedule(static)
		for (int i = 1; i < n_segment; ++i)
			offset[i] = count[i*num_threads];
		offset[n_segment] = id_end;
	} else {
		std::vector<size_t> count(n_segment, 0);

		for (auto i = id_begin; i < id_end; ++i) {
			auto key = partitioner(i);
			++count[key];
		}

		/* add initial count to this count */
		#pragma omp for schedule(static)
		for (long long int i = 0; i < n_segment; ++i)
			count[i] += offset[i + 1] - offset[i];
		
		__gnu_parallel::partial_sum(count.begin(), count.begin() + n_segment, count.begin());

		/* copy old indexes into the gaps */
		#pragma omp parallel for schedule(static)
		for (long long int i = 0; i < n_segment; ++i) {
			long long int begin = offset[i];
			long long int end = offset[i + 1];

			long long int j_offset = count[i] - end;
			count[i] -= end - begin;

			for (long long int j = begin; j < end; ++j)
				idx_out[j + j_offset] = old_idx[j];
		}

		for (auto i = id_begin; i < id_end; ++i) {
			auto key = partitioner(i);
			idx_out[--count[key]] = i;
		}

		#pragma omp parallel for schedule(static)
		for (int i = 0; i < n_segment; ++i)
			offset[i] = count[i];
		offset[n_segment] = id_end;
	}
}