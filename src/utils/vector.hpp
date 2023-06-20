#pragma once

typedef unsigned uint;

#include <iostream>
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
	template <typename T>
	class fast_vector/*numa_vector*/ {
	private:
	    mutable T* ptr = NULL;
	    mutable T* unaligned_ptr = NULL;
	    mutable size_t size_ = 0, capacity_ = 0;
	 
	public:
		template<typename Int=size_t>
	    explicit fast_vector(const Int n = 0) {
	    	resize(n);
		}

		~fast_vector() {
			if (ptr != NULL) {
				free(ptr);
				ptr = NULL;
				size_ = 0;
				capacity_ = 0;
			}
		}
	 
	    // NOT SUPPORTED !!!
	    size_t push_back(T) {
	    	exit(0);
	    	return 0;
	    }
	 
	    // function that returns the popped element
	    T pop_back() {
	    	return ptr[size_-- - 1];
	    }
	 
	    // Function that return the size of vector
	    size_t size() const {
	    	return size_;
	    }

		template<typename Int=size_t>
	    T& operator[](Int index) {
		    return *(ptr + index);
		}

		template<typename Int=size_t>
		T operator[](size_t index) const {
		    return *(ptr + index);
		}

		template<typename Int=size_t>
		T& at(const Int index) {
			if (index > size_) {
				std::cerr << "index out of bound in fast vector !\n";
				throw;
			}

		    return *(ptr + index);
		}

		template<typename Int=size_t>
		T at(const Int index) const {
			if (index > size_) {
				std::cerr << "index out of bound in fast vector !\n";
				throw;
			}

		    return *(ptr + index);
		}

		/*
		"upsize_policy" is a multiplier (>1) that forces any upsize to add a margin to avoid frequent resize.
		"downsize_policy" is a multiplier (<1) that forces a down_size to free memory only if the freed memory exceed the downsize_policy
			(to allow memory to be freed and given back to another vector).
		"min_state_size" is the minimum size of a vector, to avoid small vectors which are bound to be resized frequently.
		*/
		/// align_byte_length_ should be used to reallign the buffer, which is not yet implemented as realloc doesn't allocate.
		template<typename Int=size_t>
	    void resize(const Int n, const uint align_byte_length_=std::alignment_of<T>()) const {
	    	size_t capped_size = std::max(min_vector_size, (size_t)n); // never resize under min_vector_size

	    	if (capacity_ < capped_size || // resize if we absolutely have to because the state won't fit
	    		capped_size*upsize_policy < capacity_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)
	    		// for later allignment
	    		size_t old_size_ = size_;

	    		size_     = n;
	    		capacity_ = capped_size*upsize_policy;

	    		int offset = std::distance(unaligned_ptr, ptr);
	    		unaligned_ptr = (T*)realloc(unaligned_ptr, (capacity_ + align_byte_length_)*sizeof(T));

	    		if (unaligned_ptr == NULL)
	    			throw std::runtime_error("bad allocation in fast_vector !! size=" + std::to_string(capacity_) + "+" + std::to_string(align_byte_length_) + ", offset=" + std::to_string(offset));

	    		ptr = unaligned_ptr + offset;
	    		if (align_byte_length_ > 1) {
	    			// manual allignment:
	    			size_t allign_offset = ((size_t)ptr)%align_byte_length_;

	    			if (allign_offset != 0)
		    			// allign by rotating to the left:
		    			if (allign_offset <= offset) {
		    				std::rotate<char*>(((char*)ptr) - allign_offset, (char*)ptr, ((char*)ptr) + old_size_*sizeof(T));
		    				ptr -= allign_offset;
		    			} else {
		    				// allign by rotating to the right
		    				allign_offset = align_byte_length_ - allign_offset;
		    				std::rotate<char*>((char*)ptr, ((char*)ptr) + old_size_*sizeof(T), ((char*)ptr) + old_size_*sizeof(T) + allign_offset);
		    				ptr += allign_offset;
		    			}
	    		}
	    	} else {
	    		size_ = n;
	    	}
	    }
	 
	    // Begin iterator
	    inline T* begin() const {
	    	return ptr;
	    }
	 
	    // End iterator
	    inline T* end() const {
	    	return begin() + size_;
	    }
	};
}