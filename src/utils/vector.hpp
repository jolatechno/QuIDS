#pragma once

#include <stdexcept>
#include <iterator>     // std::iterator, std::input_iterator_tag
#include <algorithm>

/* default variables preprocessor definition:
	- "UPSIZE_POLICY" corresponds to "upsize_policy" (described in iteration_t resize operators).
	- "DOWNSIZE_POLICY" corresponds to "downsize_policy" (described in iteration_t resize operators).
	- "MIN_VECTOR_SIZE" corresponds to "min_state_size" which is the smallest size of a vector (described in iteration_t resize operators).
*/
#ifndef UPSIZE_POLICY
	#define UPSIZE_POLICY 1.1
#endif
#ifndef DOWNSIZE_POLICY
	#define DOWNSIZE_POLICY 0.85
#endif
#ifndef MIN_VECTOR_SIZE
	#define MIN_VECTOR_SIZE 1000
#endif

/// QuIDS utility function and variable namespace
namespace quids::utils {

	/// size multiplicator when upsizing, to avoid repeated upsizing.
	float upsize_policy = UPSIZE_POLICY;
	/// size multiplicator when downsize_policy, to avoid repeated upsizing. !!! upsize_policy*downsize_policy < 1 to avoid repeated resizing.
	float downsize_policy = DOWNSIZE_POLICY;
	/// minimum size a vector is allocated to (to avoid resizing at small sizes).
	size_t min_vector_size = MIN_VECTOR_SIZE;

	/// drop-in replacement for vectors, with more "efficient" memory usage and access.
	template <typename value_type>
	class fast_vector/*numa_vector*/ {
	private:
	    mutable value_type* ptr = NULL;
	    mutable value_type* unaligned_ptr = NULL;
	    mutable size_t size_ = 0;
	 
	public:
	    explicit fast_vector(size_t n = 0) {
	    	resize(n);
		}

		~fast_vector() {
			if (ptr != NULL) {
				free(ptr);
				ptr = NULL;
				size_ = 0;
			}
		}
	 
	    // NOT SUPPORTED !!!
	    size_t push_back(value_type) {
	    	exit(0);
	    	return 0;
	    }
	 
	    // function that returns the popped element
	    value_type pop_back() {
	    	return ptr[size_-- - 1];
	    }
	 
	    // Function that return the size of vector
	    size_t size() const {
	    	return size_;
	    }

	    value_type& operator[](size_t index) {
		    return *(ptr + index);
		}

		value_type operator[](size_t index) const {
		    return *(ptr + index);
		}

		/*
		"upsize_policy" is a multiplier (>1) that forces any upsize to add a margin to avoid frequent resize.
		"downsize_policy" is a multiplier (<1) that forces a down_size to free memory only if the freed memory exceed the downsize_policy
			(to allow memory to be freed and given back to another vector).
		"min_state_size" is the minimum size of a vector, to avoid small vectors which are bound to be resized frequently.
		*/
		/// align_byte_length_ should be used to reallign the buffer, which is not yet implemented as realloc doesn't allocate.
	    void resize(size_t n, const uint align_byte_length_=std::alignment_of<value_type>()) const {
	    	n = std::max(min_vector_size, n); // never resize under min_vector_size

	    	if (size_ < n || // resize if we absolutely have to because the state won't fit
	    		n*upsize_policy < size_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)
	    		size_ = n*upsize_policy + align_byte_length_;
	    		int offset = std::distance(unaligned_ptr, ptr);
	    		unaligned_ptr = (value_type*)realloc(unaligned_ptr, (size_ + offset)*sizeof(value_type));

	    		if (unaligned_ptr == NULL)
	    			throw std::runtime_error("bad allocation in fast_vector !!");

	    		ptr = unaligned_ptr + offset;
	    		if (align_byte_length_ > 1)
		    		if (NULL == std::align(align_byte_length_, size_ - align_byte_length_, (void*&)ptr, size_))
		    			throw std::runtime_error("bad alignment in fast_vector !!");
	    	}
	    }
	 
	    // Begin iterator
	    inline value_type* begin() const {
	    	return ptr;
	    }
	 
	    // End iterator
	    inline value_type* end() const {
	    	return begin() + size_;
	    }
	};
}