#include <iterator>

template <class UnsignedIntIterator1, class UnsignedIntIterator2>
void inline load_balancing_from_prefix_sum(UnsignedIntIterator1 prefixSumLoadBegin, UnsignedIntIterator1 prefixSumLoadEnd,
	UnsignedIntIterator2 workSharingIndexesBegin, UnsignedIntIterator2 workSharingIndexesEnd) {

	const unsigned long int num_segments = std::distance(workSharingIndexesBegin, workSharingIndexesEnd) - 1;

	unsigned long int w_min = *(prefixSumLoadEnd - 1) / num_segments;
	unsigned long int w_max = 2*w_min + 1;

	unsigned long int *separators = new unsigned long int[num_segments - 1]();

	/* probe function */
	auto probe_load_sharing = [&](unsigned long int wprobe) {
		/* separators */
		unsigned long int last_separator = 0;
		unsigned long int last_separator_value = 0;

		/* find separators */
		for (unsigned long int separator = 0; separator < num_segments - 1; ++separator) {
			if (*(prefixSumLoadEnd - 1) - last_separator_value > wprobe * (num_segments - separator))
				return false;

			/* dichotomie search of separator */
			last_separator = std::distance(prefixSumLoadBegin,
				std::upper_bound(prefixSumLoadBegin + last_separator, prefixSumLoadEnd, wprobe + last_separator_value)) - 1;

			/* prepare next iteration */
			last_separator_value = prefixSumLoadBegin[last_separator];
			separators[separator] = last_separator;
		}

		/* check last segment */
		if (*(prefixSumLoadEnd - 1) - last_separator_value > wprobe)
			return false;

		/* copy separators over */
		for (unsigned long int i = 0; i < num_segments - 1; ++i)
			workSharingIndexesBegin[i + 1] = separators[i] - 1;

		return true;
	};

	/* check if their is a single segment */
	if (num_segments > 1) {

		/* dichotomic search */
		// "inverse" dichotomic search to find the upper bound
		while(!probe_load_sharing(w_max)) { w_max *= 2; };

		// actual dichotomic search
		while (w_max - w_min > 1) {
			unsigned long int middle = (w_min + w_max) / 2;

			if (probe_load_sharing(middle)) {
				w_max = middle;
			} else
				w_min = middle;
		}
	}

	delete[] separators;
	*workSharingIndexesBegin = 0;
	*(workSharingIndexesEnd - 1) = std::distance(prefixSumLoadBegin, prefixSumLoadEnd) - 1;
}

template <class UnsignedIntIterator1, class UnsignedIntIterator2>
void inline indexed_load_balancing_from_prefix_sum(UnsignedIntIterator1 prefixSumLoadBegin, UnsignedIntIterator1 prefixSumLoadEnd,
	UnsignedIntIterator2 workSharingIndexesBegin, UnsignedIntIterator2 workSharingIndexesEnd) {

	/* load balance */
	load_balancing_from_prefix_sum(prefixSumLoadBegin, prefixSumLoadEnd, workSharingIndexesBegin, workSharingIndexesEnd);

	/* de-index limits from prefix sum */
	for (auto workSharingIt = workSharingIndexesBegin; workSharingIt != workSharingIndexesEnd; ++workSharingIt)
		*workSharingIt = prefixSumLoadBegin[*workSharingIt];
}