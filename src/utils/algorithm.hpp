/*
function to partition into n section according to the modulo of an array element
*/
void generalized_modulo_partition(size_t *idx_begin, size_t *idx_end, size_t const *begin, int *offset, int n_segment, int multiplier = 1) {
	/* limit values */
	size_t n_element = std::distance(idx_begin, idx_end);
	offset[0] = 0;

	if (n_segment <= 1) {
		offset[n_segment] = n_element;
		return;
	}

	/* count occurences */
	int *end = new int[n_segment]();
	#pragma omp parallel
	{
		int *count = new int[n_segment]();

		#pragma omp for schedule(static)
		for (auto it = idx_begin; it != idx_end; ++it) {
			size_t value = begin[*it];
			auto key = (value / multiplier) % n_segment;
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
			auto key = (begin[idx_begin[i]] / multiplier) % n_segment;
			if (key == partition) {
				++i;
			} else
				std::swap(idx_begin[--end[key]], idx_begin[i]);
		}
}