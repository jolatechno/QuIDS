#include <iterator>

template <class UnsignedIntIterator1, class UnsignedIntIterator2>
void inline load_balancing_from_prefix_sum(UnsignedIntIterator1 prefixSumLoadBegin, UnsignedIntIterator1 prefixSumLoadEnd,
	UnsignedIntIterator2 workSharingIndexesBegin, UnsignedIntIterator2 workSharingIndexesEnd) {

	const size_t num_segments = std::distance(workSharingIndexesBegin, workSharingIndexesEnd) - 1;

	size_t w_min = *(prefixSumLoadEnd - 1) / num_segments;
	size_t w_max = 2*w_min + 1;

	std::vector<size_t> separators(num_segments - 1, 0);

	/* probe function */
	auto probe_load_sharing = [&](size_t wprobe) {
		/* separators */
		size_t last_separator = 0;
		size_t last_separator_value = 0;

		/* find separators */
		for (size_t separator = 0; separator < num_segments - 1; ++separator) {
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
		for (size_t i = 0; i < num_segments - 1; ++i)
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
			size_t middle = (w_min + w_max) / 2;

			if (probe_load_sharing(middle)) {
				w_max = middle;
			} else
				w_min = middle;
		}
	}
	
	*workSharingIndexesBegin = 0;
	*(workSharingIndexesEnd - 1) = std::distance(prefixSumLoadBegin, prefixSumLoadEnd) - 1;
}