#pragma once

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
	#define MIN_VECTOR_SIZE 100000
#endif

// global variable definition
float upsize_policy = UPSIZE_POLICY;
float downsize_policy = DOWNSIZE_POLICY;
size_t min_vector_size = MIN_VECTOR_SIZE;

template <typename value_type>
class numa_vector {
private:
    mutable value_type* ptr = NULL;
    mutable size_t size_ = 0;
 
public:
    explicit numa_vector(size_t n = 0) {
    	resize(n);
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
	    if (index >= size_)
	        throw;
	 
	    return *(ptr + index);
	}

	value_type operator[](size_t index) const {
	    if (index >= size_)
	        throw;
	 
	    return *(ptr + index);
	}

	/*
	"upsize_policy" is a multiplier (>1) that forces any upsize to add a margin to avoid frequent resize.

	"downsize_policy" is a multiplier (<1) that forces a down_size to free memory only if the freed memory exceed the downsize_policy
		(to allow memory to be freed and given back to another vector).

	"min_state_size" is the minimum size of a vector, to avoid small vectors which are bound to be resized frequently.
	*/
    void resize(size_t n) const {
    	n = std::max(min_vector_size, n); // never resize under min_vector_size

    	if (ptr == NULL) { // just alloc if the vector is empty
    		size_ = n*upsize_policy; // resize with a margin so we don't resize too often
    		ptr = (value_type *)malloc(size_*sizeof(value_type));

    		#pragma omp for schedule(static)
    		for (size_t i = 0; i < size_; ++i)
    			volatile value_type _ = ptr[i]; // touch memory

    		return;
    	}

    	if (size_ < n || // resize if we absolutely have to because the state won't fit
    		n*upsize_policy < size_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)

    		size_t old_size = size_;
    		size_ = n*upsize_policy; // resize with a margin so we don't resize too often

#ifdef EFFICIENT_RESIZE
    		
    		size_t half_size = size_ / 2; // resize in two part to limit memory overhead

			/*
			copy the first half
			*/

			// alloc the first part
    		value_type *new_ptr = (value_type *)malloc(half_size*sizeof(value_type));

    		// copy the first part
    		#pragma omp parallel for schedule(static)
    		for (size_t i = 0; i < size_; ++i)
    			if (i < half_size)
    				if (i < old_size) {
    					new_ptr[i] = ptr[i];
    				} else
    					ptr[i] = 0; // touch memory

    		/*
			copy the second half
			*/

    		// move the second half to the first part of the old pointer
    		size_t second_half = old_size > half_size ? old_size - half_size : 0;
    		#pragma omp parallel for schedule(static)
    		for (size_t i = 0; i < second_half; ++i)
    			ptr[i] = ptr[half_size + i];

    		// realloc
    		ptr = (value_type*)realloc(ptr, second_half*sizeof(value_type));
    		new_ptr = (value_type*)realloc(new_ptr, size_*sizeof(value_type));

    		// copy the second part
    		#pragma omp parallel for schedule(static)
    		for (size_t i = 0; i < size_; ++i)
    			if (i > half_size)
    				if (i - half_size < second_half) {
    					new_ptr[i] = ptr[i - half_size];
    				} else
    					volatile value_type _ = new_ptr[i]; // touch memory

    		// free old buffer and swap them
    		free(ptr);
    		ptr = new_ptr;

#elif defined(STUPID_RESIZE)

	    	ptr = (value_type*)realloc(ptr, size_*sizeof(value_type));
	    	#pragma omp parallel for schedule(static)
	    	for (size_t i = 0; i < size_; ++i)
	    		volatile value_type _ = ptr[i]; // touch memory

#else

	    	value_type *new_ptr = (value_type *)malloc(size_*sizeof(value_type));
	    	#pragma omp parallel for schedule(static)
	    	for (size_t i = 0; i < size_; ++i)
	    		if (i < old_size) {
	    			new_ptr[i] = ptr[i];
	    		} else
	    			new_ptr[i] = 0; // touch memory

	    	// free old buffer and swap them
	    	free(ptr);
	    	ptr = new_ptr;

#endif
	    }
    }

    void iota_resize(size_t n) {
    	n = std::max(min_vector_size, n); // never resize under min_vector_size

    	if (size_ < n || // resize if we absolutely have to because the state won't fit
    		n*upsize_policy < size_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)

    		if (ptr != NULL)
				free(ptr);

    		size_ = n*upsize_policy; // resize with a margin so we don't resize too often

    		ptr = (value_type *)malloc(size_*sizeof(value_type));

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < size_; ++i)
				ptr[i] = i;
    	} else
    		// iota anyway
    		#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < size_; ++i)
				ptr[i] = i;
    }

    void zero_resize(size_t n) {
    	static value_type zero;

    	n = std::max(min_vector_size, n); // never resize under min_vector_size

    	if (size_ < n || // resize if we absolutely have to because the state won't fit
    		n*upsize_policy < size_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)

    		if (ptr != NULL)
				free(ptr);

    		size_ = n*upsize_policy; // resize with a margin so we don't resize too often

    		ptr = (value_type *)malloc(size_*sizeof(value_type));

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < size_; ++i)
				ptr[i] = 0;
    	} else
    		// iota anyway
    		#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < size_; ++i)
				ptr[i] = 0;
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