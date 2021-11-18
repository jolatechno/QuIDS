#pragma once

#include <parallel/algorithm>
#include <parallel/numeric>

#include <complex>
#include <cstddef>
#include <vector>
#include "utils/libs/robin_hood.h"

#ifndef PROBA_TYPE
	#define PROBA_TYPE double
#endif
#ifndef TOLERANCE
	#define TOLERANCE 1e-18
#endif
#ifndef SAFETY_MARGIN
	#define SAFETY_MARGIN 0.2
#endif
#ifndef SIZE_AVERAGE_PROPORTION
	#define SIZE_AVERAGE_PROPORTION 0.1
#endif
#ifndef MIN_COLLISION_SIZE
	#define MIN_COLLISION_SIZE MIN_VECTOR_SIZE
#endif
#ifndef COLLISION_TEST_PROPORTION
	#define COLLISION_TEST_PROPORTION 0.1
#endif
#ifndef COLLISION_TOLERANCE
	#define COLLISION_TOLERANCE 0.05
#endif
#ifndef LOAD_BALANCING_BUCKET_PER_THREAD
	#define LOAD_BALANCING_BUCKET_PER_THREAD 8
#endif

/*
defining openmp function's return values if openmp isn't installed or loaded
*/ 
#ifndef _OPENMP
	#define omp_set_nested(i)
	#define omp_get_thread_num() 0
	#define omp_get_num_threads() 1
#else
	#include <omp.h>
#endif

namespace iqs {
	namespace utils {
		#include "utils/load_balancing.hpp"
		#include "utils/algorithm.hpp"
		#include "utils/memory.hpp"
		#include "utils/random.hpp"
		#include "utils/vector.hpp"
	}

	/*
	global variable definition
	*/
	PROBA_TYPE tolerance = TOLERANCE;
	float safety_margin = SAFETY_MARGIN;
	size_t min_collision_size = MIN_COLLISION_SIZE;
	float collision_test_proportion = COLLISION_TEST_PROPORTION;
	float collision_tolerance = COLLISION_TOLERANCE;
	float size_average_proportion = SIZE_AVERAGE_PROPORTION;
	int load_balancing_bucket_per_thread = LOAD_BALANCING_BUCKET_PER_THREAD;

	/*
	number of threads
	*/
	const size_t num_threads = []() {
		/* get num thread */
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		return num_threads;
	}();

	/* forward typedef */
	typedef std::complex<PROBA_TYPE> mag_t;
	typedef class iteration it_t;
	typedef class symbolic_iteration sy_it_t;
	typedef class rule rule_t;
	typedef std::function<void(char* parent_begin, char* parent_end, mag_t &mag)> modifier_t;
	typedef std::function<void(int step)> debug_t;

	/* 
	rule virtual class
	*/
	class rule {
	public:
		rule() {};
		virtual inline void get_num_child(char const *parent_begin, char const *parent_end, uint32_t &num_child, size_t &max_child_size) const = 0;
		virtual inline void populate_child(char const *parent_begin, char const *parent_end, char* const child_begin, uint32_t const child_id, size_t &size, mag_t &mag) const = 0;
		virtual inline size_t hasher(char const *parent_begin, char const *parent_end) const { //can be overwritten
			return std::hash<std::string_view>()(std::string_view(parent_begin, std::distance(parent_begin, parent_end)));
		}
	};

	/*
	iteration class
	*/
	class iteration {
		friend symbolic_iteration;
		friend size_t inline get_max_num_object(it_t const &next_iteration, it_t const &last_iteration, sy_it_t const &symbolic_iteration);
		friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration, debug_t mid_step_function);  
		friend void inline simulate(it_t &iteration, modifier_t const rule);

	protected:
		utils::numa_vector<mag_t> magnitude;
		utils::numa_vector<char> objects;
		utils::numa_vector<size_t> object_begin;
		mutable utils::numa_vector<uint32_t> num_childs;

		void inline resize(size_t num_object) const {
			magnitude.resize(num_object);
			num_childs.zero_resize(num_object + 1);
			object_begin.resize(num_object + 1);
		}
		void inline allocate(size_t size) const {
			objects.resize(size);
		}

		void generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function) const;
		void apply_modifier(modifier_t const rule);
		void normalize();

		long long int memory_size = 2*sizeof(PROBA_TYPE) + sizeof(size_t) + sizeof(uint32_t);

	public:
		size_t num_object = 0;
		PROBA_TYPE total_proba = 1;

		iteration() {
			resize(0);
			allocate(0);
			object_begin[0] = 0;
			num_childs[0] = 0;
		}
		iteration(char* object_begin_, char* object_end_) : iteration() {
			append(object_begin_, object_end_);
		}
		void append(char const *object_begin_, char const *object_end_, mag_t const mag=1) {
			size_t offset = object_begin[num_object];
			size_t size = std::distance(object_begin_, object_end_);

			resize(++num_object);
			allocate(offset + size);

			for (size_t i = 0; i < size; ++i)
				objects[offset + i] = object_begin_[i];

			magnitude[num_object - 1] = mag;
			object_begin[num_object] = offset + size;
		}
		void pop(size_t n=1, bool normalize_=true) {
			if (n < 1)
				return;

			num_object -= n;
			allocate(object_begin[num_object]);
			resize(num_object);

			if (normalize_) normalize();
		}
		template<class T>
		T average_value(std::function<T(char const *object_begin, char const *object_end)> const &observable) const {
			T avg = 0;

			#pragma omp parallel
			{	
				/* compute average per thread */
				T local_avg = 0;
				#pragma omp for schedule(static)
				for (size_t oid = 0; oid < num_object; ++oid) {
					size_t size;
					std::complex<PROBA_TYPE> mag;

					/* get object and accumulate */
					char const *this_object_begin;
					get_object(oid, this_object_begin, size, mag);
					local_avg += observable(this_object_begin, this_object_begin + size) * std::norm(mag);
				}

				/* accumulate thread averages */
				#pragma omp critical
				avg += local_avg;
			}

			return avg;
		}
		void get_object(size_t const object_id, char *& object_begin_, size_t &object_size, mag_t *&mag) {
			size_t this_object_begin = object_begin[object_id];
			object_size = object_begin[object_id + 1] - this_object_begin;
			mag = &magnitude[object_id];
			object_begin_ = objects.begin() + this_object_begin;
		}
		void get_object(size_t const object_id, char const *& object_begin_, size_t &object_size, mag_t &mag) const {
			size_t this_object_begin = object_begin[object_id];
			object_size = object_begin[object_id + 1] - this_object_begin;
			mag = magnitude[object_id];
			object_begin_ = objects.begin() + this_object_begin;
		}
	};

	/*
	symboluc iteration class
	*/
	class symbolic_iteration {
		friend iteration;
		friend size_t inline get_max_num_object(it_t const &next_iteration, it_t const &last_iteration, sy_it_t const &symbolic_iteration);
		friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration, debug_t mid_step_function); 

	protected:
		//tbb::concurrent_hash_map<size_t, size_t> elimination_map;
		std::vector<robin_hood::unordered_map<size_t, size_t>> elimination_maps = std::vector<robin_hood::unordered_map<size_t, size_t>>(num_threads);
		std::vector<char*> placeholder = std::vector<char*>(num_threads, NULL);

		utils::numa_vector<mag_t> magnitude;
		utils::numa_vector<size_t> next_oid;
		utils::numa_vector<size_t> size;
		utils::numa_vector<size_t> hash;
		utils::numa_vector<size_t> parent_oid;
		utils::numa_vector<uint32_t> child_id;
		utils::numa_vector<bool> is_unique;
		utils::numa_vector<double> random_selector;

		void inline resize(size_t num_object) {
			magnitude.resize(num_object);
			next_oid.iota_resize(num_object);
			size.zero_resize(num_object);
			hash.zero_resize(num_object);
			parent_oid.zero_resize(num_object);
			child_id.zero_resize(num_object);
			is_unique.zero_resize(num_object);
			random_selector.zero_resize(num_object);
		}
		void inline reserve(size_t max_size) {
			#pragma omp parallel
			{
				auto &buffer = placeholder[omp_get_thread_num()];
				if (buffer == NULL)
					free(buffer);
				buffer = new char[max_size];
				for (size_t i = 0; i < max_size; ++i) ((volatile char*)buffer)[i] = 0; // touch
			}
		}

		void compute_collisions();
		void finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, debug_t mid_step_function);

		long long int memory_size = 1 + 2*sizeof(PROBA_TYPE) + 6*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);

	public:
		size_t num_object = 0;
		size_t num_object_after_interferences = 0;

		symbolic_iteration() {}
	};

	/*
	for memory managment
	*/
	size_t inline get_max_num_object(it_t const &next_iteration, it_t const &last_iteration, sy_it_t const &symbolic_iteration) {
		// get the free memory and the total amount of memory...
		size_t free_mem;
		utils::get_free_mem(free_mem);

		// get the total memory
		size_t total_useable_memory = next_iteration.objects.size() + last_iteration.objects.size() + // size of objects
			last_iteration.magnitude.size()*last_iteration.memory_size + next_iteration.magnitude.size()*next_iteration.memory_size + // size of properties
			symbolic_iteration.magnitude.size()*symbolic_iteration.memory_size + // size of symbolic properties
			free_mem; // free memory

		// compute average object size
		size_t iteration_size_per_object = 0;

		// compute the average size of an object for the next iteration:
		size_t test_size = std::max((size_t)1, (size_t)(size_average_proportion*symbolic_iteration.num_object_after_interferences));
		#pragma omp parallel for reduction(+:iteration_size_per_object)
		for (size_t oid = 0; oid < test_size; ++oid)
			iteration_size_per_object += symbolic_iteration.size[oid];
		iteration_size_per_object /= test_size;

		// add the cost of the symbolic iteration in itself
		iteration_size_per_object += symbolic_iteration.memory_size*symbolic_iteration.num_object/last_iteration.num_object/2; // size for symbolic iteration
		
		// and the constant size per object
		iteration_size_per_object += next_iteration.memory_size;

		// and the cost of unused space
		iteration_size_per_object *= utils::upsize_policy;

		return total_useable_memory / iteration_size_per_object * (1 - safety_margin);
	}

	/*
	simulation function
	*/
	void inline simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration, debug_t mid_step_function=[](int){}) {
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
		symbolic_iteration.compute_collisions();
		symbolic_iteration.finalize(rule, iteration, iteration_buffer, mid_step_function);
		iteration_buffer.normalize();

		mid_step_function(8);
		
		std::swap(iteration_buffer, iteration);
	}
	void inline simulate(it_t &iteration, modifier_t const rule) {
		iteration.apply_modifier(rule);
	}

	/*
	apply modifier
	*/
	void iteration::apply_modifier(modifier_t const rule) {
		#pragma omp parallel for schedule(static)
		for (size_t oid = 0; oid < num_object; ++oid)
			/* generate graph */
			rule(objects.begin() + object_begin[oid],
				objects.begin() + object_begin[oid + 1],
				magnitude[oid]);
	}

	/*
	generate symbolic iteration
	*/
	void iteration::generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function=[](int){}) const {
		if (num_object == 0) {
			symbolic_iteration.num_object = 0;
			return;
		}

		size_t max_size;

		mid_step_function(0);

		/* !!!!!!!!!!!!!!!!
		step (1)
		 !!!!!!!!!!!!!!!! */

		#pragma omp parallel for schedule(static) reduction(max:max_size)
		for (size_t oid = 0; oid < num_object; ++oid) {
			size_t size;
			rule->get_num_child(objects.begin() + object_begin[oid],
				objects.begin() + object_begin[oid + 1],
				num_childs[oid + 1], size);
			max_size = std::max(max_size, size);
		}

		mid_step_function(1);

		/* !!!!!!!!!!!!!!!!
		step (2)
		 !!!!!!!!!!!!!!!! */

		__gnu_parallel::partial_sum(num_childs.begin() + 1, num_childs.begin() + num_object + 1, num_childs.begin() + 1);
		symbolic_iteration.num_object = num_childs[num_object];

		/* resize symbolic iteration */
		symbolic_iteration.resize(symbolic_iteration.num_object);
		symbolic_iteration.reserve(max_size);
		
		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();

			#pragma omp for schedule(static)
			for (size_t oid = 0; oid < num_object; ++oid) {
				/* assign parent ids and child ids for each child */
				std::fill(symbolic_iteration.parent_oid.begin() + num_childs[oid],
					symbolic_iteration.parent_oid.begin() + num_childs[oid + 1],
					oid);
				std::iota(symbolic_iteration.child_id.begin() + num_childs[oid],
					symbolic_iteration.child_id.begin() + num_childs[oid + 1],
					0);
			}

			#pragma omp single
			mid_step_function(2);

			/* !!!!!!!!!!!!!!!!
			step (3)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for schedule(static)
			for (size_t oid = 0; oid < symbolic_iteration.num_object; ++oid) {
				auto id = symbolic_iteration.parent_oid[oid];

				/* generate graph */
				symbolic_iteration.magnitude[oid] = magnitude[id];
				rule->populate_child(objects.begin() + object_begin[id],
					objects.begin() + object_begin[id + 1],
					symbolic_iteration.placeholder[thread_id], symbolic_iteration.child_id[oid],
					symbolic_iteration.size[oid], symbolic_iteration.magnitude[oid]);

				/* compute hash */
				symbolic_iteration.hash[oid] = rule->hasher(symbolic_iteration.placeholder[thread_id],
					symbolic_iteration.placeholder[thread_id] + symbolic_iteration.size[oid]);
			}
		}

		mid_step_function(3);
	}

	/*
	compute interferences
	*/
	void symbolic_iteration::compute_collisions() {
		if (num_object == 0) {
			num_object_after_interferences = 0;
			return;
		}

		/* !!!!!!!!!!!!!!!!
		step (4)
		 !!!!!!!!!!!!!!!! */

		bool fast = false;
		bool skip_test = collision_test_proportion == 0 || collision_tolerance == 0 || num_object < min_collision_size;
		size_t test_size = skip_test ? 0 : num_object*collision_test_proportion;
		const int num_bucket = num_threads > 1 ? load_balancing_bucket_per_thread*num_threads : 1;

		int *modulo_offset = new int[num_bucket + 1];
		int *load_balancing_begin = new int[num_threads + 1];

		if (!skip_test) {
			/* partition to limit collisions */
			utils::generalized_modulo_partition(next_oid.begin(), next_oid.begin() + test_size,
				hash.begin(), modulo_offset, num_bucket);
			utils::load_balancing_from_prefix_sum(modulo_offset, modulo_offset + num_bucket,
				load_balancing_begin, load_balancing_begin + num_threads + 1);

			size_t size_after_insertion = 0;
			#pragma omp parallel
			{
				int thread_id = omp_get_thread_num();

				size_t begin = modulo_offset[load_balancing_begin[thread_id]];
				size_t end = modulo_offset[load_balancing_begin[thread_id + 1]];

				auto &elimination_map = elimination_maps[thread_id];
				elimination_map.reserve(end - begin);

				for (size_t i = begin; i < end; ++i) {
					size_t oid = next_oid[i];

					/* accessing key */
					auto [it, unique] = elimination_map.insert({hash[oid], oid});
					if (unique) {
						is_unique[oid] = true; /* keep this graph */
					} else {
						/* if it exist add the probabilities */
						magnitude[it->second] += magnitude[oid];

						/* discard this graph */
						is_unique[oid] = false;
					}
				}

				#pragma omp atomic
				size_after_insertion += elimination_map.size();
			}

			/* check if we should continue */
			fast = test_size - size_after_insertion < test_size*collision_tolerance;
			if (fast)
				/* get all unique graphs with a non zero probability */
				#pragma omp parallel for schedule(static)
				for (size_t oid = test_size; oid < num_object; ++oid)
					is_unique[oid] = true;
		}

		if (!fast) {
			/* partition to limit collisions */
			utils::generalized_modulo_partition(next_oid.begin() + test_size, next_oid.begin() + num_object,
				hash.begin(), modulo_offset, num_bucket);
			utils::load_balancing_from_prefix_sum(modulo_offset, modulo_offset + num_bucket,
				load_balancing_begin, load_balancing_begin + num_threads + 1);

			#pragma omp parallel
			{
				int thread_id = omp_get_thread_num();

				size_t begin = modulo_offset[load_balancing_begin[thread_id]] + test_size;
				size_t end = modulo_offset[load_balancing_begin[thread_id + 1]] + test_size;

				auto &elimination_map = elimination_maps[thread_id];
				elimination_map.reserve(end - begin + elimination_map.size());
				
				for (size_t i = begin; i < end; ++i) {
					size_t oid = next_oid[i];
					
					/* accessing key */
					auto [it, unique] = elimination_map.insert({hash[oid], oid});
					if (unique) {
						is_unique[oid] = true; /* keep this graph */
					} else {
						/* if it exist add the probabilities */
						magnitude[it->second] += magnitude[oid];

						/* discard this graph */
						is_unique[oid] = false;
					}
				}
			}
		}

		/* get all unique graphs with a non zero probability */
		auto partitioned_it = __gnu_parallel::partition(next_oid.begin(), next_oid.begin() + num_object,
			[&](size_t const &oid) {
				/* check if graph is unique */
				if (!is_unique[oid])
					return false;

				/* check for zero probability */
				return std::norm(magnitude[oid]) > tolerance;
			});
		num_object_after_interferences = std::distance(next_oid.begin(), partitioned_it);
		
		#pragma omp parallel	
		elimination_maps[omp_get_thread_num()].clear();
	}

	/*
	finalize iteration
	*/
	void symbolic_iteration::finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, debug_t mid_step_function=[](int){}) {
		if (num_object_after_interferences == 0) {
			next_iteration.num_object = 0;
			return;
		}
		
		mid_step_function(4);

		/* !!!!!!!!!!!!!!!!
		step (5)
		 !!!!!!!!!!!!!!!! */

		long long int max_num_object = get_max_num_object(next_iteration, last_iteration, *this) / 2;
		max_num_object = std::max(max_num_object, (long long int)utils::min_vector_size);

		if (num_object_after_interferences > max_num_object) {

			/* generate random selectors */
			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < num_object_after_interferences; ++i)  {
				size_t oid = next_oid[i];

				double random_number = utils::unfiorm_from_hash(hash[oid]); //random_generator();
				random_selector[oid] = std::log( -std::log(1 - random_number) / std::norm(magnitude[oid]));
			}

			/* select graphs according to random selectors */
			__gnu_parallel::nth_element(next_oid.begin(), next_oid.begin() + max_num_object, next_oid.begin() + num_object_after_interferences,
			[&](size_t const &oid1, size_t const &oid2) {
				return random_selector[oid1] < random_selector[oid2];
			});

			next_iteration.num_object = max_num_object;
		} else
			next_iteration.num_object = num_object_after_interferences;

		mid_step_function(5);

		/* !!!!!!!!!!!!!!!!
		step (6)
		 !!!!!!!!!!!!!!!! */

		/* sort to make memory access more continuous */
		__gnu_parallel::sort(next_oid.begin(), next_oid.begin() + next_iteration.num_object);

		/* !!!!!!!!!!!!!!!!!!!! debug !!!!!!!!!!!!!!!!!!!! */
		mid_step_function(6);

		/* resize new step variables */
		next_iteration.resize(next_iteration.num_object);

		/* !!!!!!!!!!!!!!!!!!!! debug !!!!!!!!!!!!!!!!!!!! */
		mid_step_function(7);
				
		/* prepare for partial sum */
		#pragma omp parallel for schedule(static)
		for (size_t oid = 0; oid < next_iteration.num_object; ++oid) {
			auto id = next_oid[oid];

			/* assign magnitude and size */
			next_iteration.object_begin[oid + 1] = size[id];
			next_iteration.magnitude[oid] = magnitude[id];
		}

		/* !!!!!!!!!!!!!!!!!!!! debug !!!!!!!!!!!!!!!!!!!! */
		mid_step_function(8);

		__gnu_parallel::partial_sum(next_iteration.object_begin.begin() + 1,
			next_iteration.object_begin.begin() + next_iteration.num_object + 1,
			next_iteration.object_begin.begin() + 1);

		next_iteration.allocate(next_iteration.object_begin[next_iteration.num_object]);

		mid_step_function(9 /*6*/);

		/* !!!!!!!!!!!!!!!!
		step (7)
		 !!!!!!!!!!!!!!!! */

		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();
			mag_t mag_;
			size_t size_;

			#pragma omp for schedule(static)
			for (size_t oid = 0; oid < next_iteration.num_object; ++oid) {
				auto id = next_oid[oid];
				auto this_parent_oid = parent_oid[id];
				
				rule->populate_child(last_iteration.objects.begin() + last_iteration.object_begin[this_parent_oid],
					last_iteration.objects.begin() + last_iteration.object_begin[this_parent_oid + 1],
					next_iteration.objects.begin() + next_iteration.object_begin[oid],
					child_id[id],
					size_, mag_);
			}
		}
		
		mid_step_function(7);
	}

	/*
	normalize
	*/
	void iteration::normalize() {
		/* !!!!!!!!!!!!!!!!
		step (8)
		 !!!!!!!!!!!!!!!! */

		total_proba = 0;

		#pragma omp parallel
		{
			#pragma omp for reduction(+:total_proba)
			for (size_t oid = 0; oid < num_object; ++oid)
				total_proba += std::norm(magnitude[oid]);

			PROBA_TYPE normalization_factor = std::sqrt(total_proba);

			if (normalization_factor != 1)
				#pragma omp for
				for (size_t oid = 0; oid < num_object; ++oid)
					magnitude[oid] /= normalization_factor;
		}
		
	}
}