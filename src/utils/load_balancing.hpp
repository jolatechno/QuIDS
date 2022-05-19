#pragma once

#include <iterator>

#ifdef VALIDATE_CCP
#include <iostream>
#endif

/// QuIDS utility function and variable namespace
namespace quids::utils {

	/// simple CCP load balacing implementation.
	template <class UnsignedIntIterator1, class UnsignedIntIterator2>
	void inline load_balancing_from_prefix_sum(const UnsignedIntIterator1 prefixSumLoadBegin, const UnsignedIntIterator1 prefixSumLoadEnd,
		UnsignedIntIterator2 workSharingIndexesBegin, UnsignedIntIterator2 workSharingIndexesEnd) {

		const size_t num_segments = std::distance(workSharingIndexesBegin, workSharingIndexesEnd) - 1;
		const size_t num_buckets = std::distance(prefixSumLoadBegin, prefixSumLoadEnd) - 1;
		
		std::fill(workSharingIndexesBegin, workSharingIndexesEnd - 1, 0);
		workSharingIndexesBegin[num_segments] = num_buckets;

		size_t w_min = prefixSumLoadBegin[num_buckets]/num_segments;
		size_t w_max = 2*w_min + 1;

		std::vector<size_t> separators(num_segments - 1, 0);

#ifdef VALIDATE_CCP
		const auto check_partition = [&](size_t wprobe) {
			const auto print_debug = [&]() {
				std::cerr << "\t";
				for (int i = 0; i <= num_segments; ++i)
					std::cerr << prefixSumLoadBegin[workSharingIndexesBegin[i]] << "[" << workSharingIndexesBegin[i] << "], ";
				std::cerr << "\n";
			};


			for (int i = 0; i < num_segments; ++i) {
				if (prefixSumLoadBegin[workSharingIndexesBegin[i + 1]] - prefixSumLoadBegin[workSharingIndexesBegin[i]] > wprobe) {
					std::cerr << "CCP failed (too big of a gap) at " << i << "/" << num_segments << "\n";
					print_debug();
					throw;
				}
				if (workSharingIndexesBegin[i + 1] + 1 < num_buckets)
					if (prefixSumLoadBegin[workSharingIndexesBegin[i + 1] + 1] - prefixSumLoadBegin[workSharingIndexesBegin[i]] < wprobe) {
						std::cerr << "CCP failed (too small of a gap) at " << i << "/" << num_segments << "\n";
						print_debug();
						throw;
					}
			}
		};
#endif

		/* probe function */
		auto probe_load_sharing = [&](size_t wprobe) {
			if (wprobe == 0 || wprobe >= prefixSumLoadBegin[num_buckets])
				return true;

			/* separators */
			size_t last_separator = 0;
			size_t last_separator_value = 0;

			/* find separators */
			for (size_t separator = 1; separator < num_segments; ++separator) {
				/* dichotomie search of separator */
				separators[separator - 1] = std::distance(prefixSumLoadBegin,
					std::lower_bound(prefixSumLoadBegin + last_separator, prefixSumLoadEnd,
						last_separator_value + wprobe)) - 1;

				/* prepare next iteration */
				last_separator = separators[separator - 1];
				last_separator_value = prefixSumLoadBegin[last_separator];


				if (prefixSumLoadBegin[num_buckets] - last_separator_value > wprobe*(num_segments - separator))
					return false;
			}

			/* copy separators over */
			for (size_t i = 0; i < num_segments - 1; ++i)
				workSharingIndexesBegin[i + 1] = separators[i];

			return true;
		};

		// don't continue if their is a single segment
		if (num_segments > 1) {
			// "inverse" dichotomic search to find the lower bound
			while(w_min != 0 && probe_load_sharing(w_min)) { w_min /= 2; };

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
	}
}