#pragma once

#include <iterator>     // std::iterator, std::input_iterator_tag

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
    mutable value_type* ptr = 0;
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
    	static value_type zero;

    	n = std::max(min_vector_size, n); // never resize under min_vector_size

    	if (size_ < n || // resize if we absolutely have to because the state won't fit
    		n*upsize_policy < size_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)

    		n *= upsize_policy; // resize with a margin so we don't resize too often

    		value_type* new_ptr = (value_type*)(new char[n*sizeof(value_type)]);

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < n; ++i)
				new_ptr[i] = i < size_ ? ptr[i] : zero;

			if (ptr != 0)
				free(ptr);

			ptr = new_ptr;
			size_ = n;
    	}
    }

    void iota_resize(size_t n) {
    	n = std::max(min_vector_size, n); // never resize under min_vector_size

    	if (size_ < n || // resize if we absolutely have to because the state won't fit
    		n*upsize_policy < size_*downsize_policy) { // resize if the size we resize to is small enough (to free memory)

    		n *= upsize_policy; // resize with a margin so we don't resize too often

    		value_type* new_ptr = (value_type*)(new char[n*sizeof(value_type)]);

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < n; ++i)
				new_ptr[i] = i;

			if (ptr != 0)
				free(ptr);

			ptr = new_ptr;
			size_ = n;
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

    		n *= upsize_policy; // resize with a margin so we don't resize too often

    		value_type* new_ptr = (value_type*)(new char[n*sizeof(value_type)]);

			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < n; ++i)
				new_ptr[i] = zero;

			if (ptr != 0)
				free(ptr);

			ptr = new_ptr;
			size_ = n;
    	} else
    		// iota anyway
    		#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < size_; ++i)
				ptr[i] = i;
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