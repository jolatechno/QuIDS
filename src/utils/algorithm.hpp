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
/*void generalized_modulo_partition_power_of_two(size_t *idx_begin, size_t *idx_end, size_t const *begin, int *offset, int n_segment, int shift = 0) {
	/* limit values */
	/*size_t n_element = std::distance(idx_begin, idx_end);
	offset[0] = 0;
	offset[n_segment] = n_element;

	if (n_segment == 1)
		return;


	const size_t bitmask = n_segment - 1;

	std::function<void(int,int)> const recursions = [&](int lower, int upper) {
		const int middle = (lower + upper) / 2;

		auto partitioned_it = __gnu_parallel::partition(idx_begin + offset[lower], idx_begin + offset[upper],
			[&](const size_t oid) {
				return (begin[oid] & bitmask) < middle;
			});
		offset[middle] = std::distance(idx_begin, partitioned_it);
		
		if (upper - lower > 2) {
			#pragma omp task
			recursions(lower, middle);

			#pragma omp task
			recursions(middle, upper);
		}
	};

	omp_set_nested(1);

	#pragma omp parallel
	#pragma omp single
	recursions(0, n_segment);
}

	/* count occurences */
	/*int *end = new int[n_segment]();
	#pragma omp parallel
	{
		int *count = new int[n_segment]();

		#pragma omp for schedule(static)
		for (auto it = idx_begin; it != idx_end; ++it) {
			size_t value = begin[*it];
			auto key = (value >> shift) & bitmask;
			++count[key];
		}

		/* global reduction */
		/*for (int i = 0; i < n_segment; ++i)
			#pragma omp atomic
			end[i] += count[i];
	}

	/* compute the end of each segment */
	/*__gnu_parallel::partial_sum(end, end + n_segment, offset + 1);
	for (int i = 0; i < n_segment; ++i)
		end[i] = offset[i + 1];

	/* move elements */
	/*int partition = 0;
	for (size_t i = 0; i < n_element;)
		if (i >= end[partition]) {
			i = offset[++partition];
		} else {
			auto key = (begin[idx_begin[i]] >> shift) & bitmask;
			if (key == partition) {
				++i;
			} else
				std::swap(idx_begin[--end[key]], idx_begin[i]);
		}
}*/

void generalized_modulo_partition_power_of_two(size_t *idx_begin, size_t *idx_end, size_t const *begin, int *offset, int n_segment) {
	/* limit values */
	size_t n_element = std::distance(idx_begin, idx_end);
	offset[0] = 0;

	if (n_segment <= 1) {
		offset[n_segment] = n_element;
		return;
	}

	const size_t bitmask = n_segment - 1;

	/* count occurences */
	int *end = new int[n_segment]();
	#pragma omp parallel
	{
		int *count = new int[n_segment]();

		#pragma omp for schedule(static)
		for (auto it = idx_begin; it != idx_end; ++it) {
			size_t value = begin[*it];
			auto key = value & bitmask;
			++count[key];
		}

		/* global reduction */
		for (int i = 0; i < n_segment; ++i)
			#pragma omp atomic
			end[i] += count[i];
	}

	/* compute the end of each segment */
	__gnu_parallel::partial_sum(end, end + n_segment, offset + 1);
	for (int i = 0; i < n_segment; ++i)
		end[i] = offset[i + 1];

	/* move elements */
	int partition = 0;
	for (size_t i = 0; i < n_element;)
		if (i >= end[partition]) {
			i = offset[++partition];
		} else {
			auto key = begin[idx_begin[i]] & bitmask;
			if (key == partition) {
				++i;
			} else
				std::swap(idx_begin[--end[key]], idx_begin[i]);
		}
}