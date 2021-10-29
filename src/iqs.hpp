#pragma once

#include <parallel/algorithm>
#include <parallel/numeric>

#include <cstddef>
#include <vector>
#include <tbb/concurrent_hash_map.h> // For concurrent hash map.

#ifndef PROBA_TYPE
	#define PROBA_TYPE double
#endif
#ifndef TOLERANCE
	#define TOLERANCE 1e-18
#endif
#ifndef SAFETY_MARGIN
	#define SAFETY_MARGIN 0.2
#endif
#ifndef COLLISION_TEST_PROPORTION
	#define COLLISION_TEST_PROPORTION 0.1
#endif
#ifndef COLLISION_TOLERANCE
	#define COLLISION_TOLERANCE 0.05
#endif

/*
defining openmp function's return values if openmp isn't installed or loaded
*/ 
#ifndef _OPENMP
	#define omp_set_nested(i)
	#define omp_get_thread_num() 0
	#define omp_get_num_thread() 1
#else
	#include <omp.h>
#endif

namespace iqs {
	namespace utils {
		#include "utils/complex.hpp"
		#include "utils/load_balancing.hpp"
		#include "utils/memory.hpp"
		#include "utils/random.hpp"
		#include "utils/vector.hpp"
	}
	

	/*
	global variable definition
	*/
	namespace {
		PROBA_TYPE tolerance = TOLERANCE;
		float safety_margin = SAFETY_MARGIN;
		float collision_test_proportion = COLLISION_TEST_PROPORTION;
		float collision_tolerance = COLLISION_TOLERANCE;
	}
	
	/*
	global variable setters
	*/
	void set_tolerance(PROBA_TYPE val) { tolerance = val; }
	void set_safety_margin(float val) { safety_margin = val; }
	void set_collision_test_proportion(float val) { collision_test_proportion = val; }
	void set_collision_tolerance(float val) { collision_tolerance = val; }

	/*
	number of threads
	*/
	const size_t num_threads = []() {
		/* allow nested parallism for __gnu_parallel inside omp single */
		omp_set_nested(3);

		/* get num thread */
		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		return num_threads;
	}();

	/* forward typedef */
	typedef class iteration it_t;
	typedef class symbolic_iteration sy_it_t;
	typedef class rule rule_t;
	typedef std::function<void(char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag)> modifier_t;
	typedef std::function<void(int step)> debug_t;

	/* 
	rule virtual class
	*/
	class rule {
	public:
		rule() {};
		virtual inline void get_num_child(char* parent_begin, char* parent_end, uint32_t &num_child, size_t &max_child_size) const = 0;
		virtual inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const = 0;
		virtual inline size_t hasher(char* parent_begin, char* parent_end) const { //can be overwritten
			return std::hash<std::string_view>()(std::string_view(parent_begin, parent_end));
		}
	};

	/*
	iteration class
	*/
	class iteration {
		friend symbolic_iteration;

	//protected:
	public:
		size_t num_object = 0;
		PROBA_TYPE total_proba = 1;

		utils::numa_vector<PROBA_TYPE> real, imag;
		utils::numa_vector<char> objects;
		utils::numa_vector<size_t> object_begin;

	private:
		mutable utils::numa_vector<uint32_t> num_childs;

		void inline resize(size_t num_object) const {
			real.resize(num_object);
			imag.resize(num_object);
			num_childs.resize(num_object + 1);
			object_begin.resize(num_object + 1);
		}
		void inline allocate(size_t size) const {
			objects.resize(size);
		}

	public:
		iteration() {
			resize(0);
			allocate(0);
			object_begin[0] = 0;
			num_childs[0] = 0;
		}
		iteration(char* object_begin_, char* object_end_) : iteration() {
			append(object_begin_, object_end_);
		}
		void append(char* object_begin_, char* object_end_, PROBA_TYPE real_ = 1, PROBA_TYPE imag_ = 0) {
			size_t offset = object_begin[num_object];
			size_t size = std::distance(object_begin_, object_end_);

			resize(++num_object);
			allocate(offset + size);

			for (size_t i = 0; i < size; ++i)
				objects[offset + i] = object_begin_[i];

			real[num_object - 1] = real_; imag[num_object - 1] = imag_;
			object_begin[num_object] = offset + size;
		}
		void generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function) const;
		void apply_modifier(modifier_t const rule);
		void normalize();
	};

	/*
	symboluc iteration class
	*/
	class symbolic_iteration {
		friend iteration;

	//protected:
	public:
		size_t num_object = 0;
		size_t num_object_after_interferences = 0;

	private:
		tbb::concurrent_hash_map<size_t, size_t> elimination_map;
		std::vector<char*> placeholder = std::vector<char*>(num_threads, NULL);

		utils::numa_vector<PROBA_TYPE> real, imag;
		utils::numa_vector<size_t> next_gid;
		utils::numa_vector<size_t> size;
		utils::numa_vector<size_t> hash;
		utils::numa_vector<size_t> parent_gid;
		utils::numa_vector<uint32_t> child_id;
		utils::numa_vector<bool> is_unique;
		utils::numa_vector<double> random_selector;

		void inline resize(size_t num_object) {
			real.resize(num_object);
			imag.resize(num_object);
			next_gid.iota_resize(num_object);
			size.zero_resize(num_object);
			hash.zero_resize(num_object);
			parent_gid.resize(num_object);
			child_id.resize(num_object);
			is_unique.resize(num_object);
			random_selector.zero_resize(num_object);
		}
		void inline reserve(size_t max_size) {
			#pragma omp parallel
			{
				auto &buffer = placeholder[omp_get_thread_num()];
				if (buffer == NULL)
					free(buffer);
				buffer = new char[max_size];
				for (size_t i = 0; i < max_size; ++i) buffer[i] = 0; // touch
			}
		}

	public:
		symbolic_iteration() {}
		void compute_collisions();
		void finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, debug_t mid_step_function);
	};

	/*
	for memory managment
	*/
	long long int inline get_additional_max_num_object(it_t const &last_iteration, sy_it_t const &symbolic_iteration) {
		// get the free memory and the total amount of memory...
		long long int total_memory, free_mem;
		utils::get_mem_usage_and_free_mem(total_memory, free_mem);

		// and according to the "safety_margin" (a proportion of total memory) compute the total delta between the amount free memory and the target
		long int mem_difference = free_mem - total_memory*safety_margin;

		static long long int iteration_size = 2*sizeof(PROBA_TYPE) + sizeof(size_t) + sizeof(uint32_t);
		static long long int symbolic_iteration_size = 1 + 2*sizeof(PROBA_TYPE) + 6*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);

		/* compute average object size */
		long long int iteration_size_per_object = last_iteration.object_begin[last_iteration.num_object]; // size for all object buffer
		iteration_size_per_object += symbolic_iteration.num_object*symbolic_iteration_size; // size for symbolic iteration
		iteration_size_per_object /= last_iteration.num_object; // divide to get size per object
		iteration_size_per_object += iteration_size; // add constant size per object

		return mem_difference / iteration_size_per_object;
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
		for (size_t gid = 0; gid < num_object; ++gid)
			/* generate graph */
			rule(objects.begin() + object_begin[gid],
				objects.begin() + object_begin[gid + 1],
				real[gid], imag[gid]);
	}

	/*
	generate symbolic iteration
	*/
	void iteration::generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function=[](int){}) const {
		size_t max_size = 0;

		mid_step_function(0);

		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();

			/* !!!!!!!!!!!!!!!!
			step (1)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for schedule(static) reduction(max:max_size)
			for (size_t gid = 0; gid < num_object; ++gid) {
				size_t size;
				rule->get_num_child(objects.begin() + object_begin[gid],
					objects.begin() + object_begin[gid + 1],
					num_childs[gid + 1], size);
				max_size = std::max(max_size, size);
			}

			#pragma omp single
			mid_step_function(1);

			/* !!!!!!!!!!!!!!!!
			step (2)
			 !!!!!!!!!!!!!!!! */

			#pragma omp single
			{
				__gnu_parallel::partial_sum(num_childs.begin() + 1, num_childs.begin() + num_object + 1, num_childs.begin() + 1);
				symbolic_iteration.num_object = num_childs[num_object];

				/* resize symbolic iteration */
				symbolic_iteration.resize(symbolic_iteration.num_object);
				symbolic_iteration.reserve(max_size);
			}

			#pragma omp for schedule(static)
			for (size_t gid = 0; gid < num_object; ++gid) {
				/* assign parent ids and child ids for each child */
				std::fill(symbolic_iteration.parent_gid.begin() + num_childs[gid],
					symbolic_iteration.parent_gid.begin() + num_childs[gid + 1],
					gid);
				std::iota(symbolic_iteration.child_id.begin() + num_childs[gid],
					symbolic_iteration.child_id.begin() + num_childs[gid + 1],
					0);
			}

			#pragma omp single
			mid_step_function(2);

			/* !!!!!!!!!!!!!!!!
			step (3)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for schedule(static)
			for (size_t gid = 0; gid < symbolic_iteration.num_object; ++gid) {
				auto id = symbolic_iteration.parent_gid[gid];

				/* generate graph */
				symbolic_iteration.real[gid] = real[id];
				symbolic_iteration.imag[gid] = imag[id];
				char* end = rule->populate_child(objects.begin() + object_begin[id],
					objects.begin() + object_begin[id + 1],
					symbolic_iteration.child_id[gid],
					symbolic_iteration.real[gid], symbolic_iteration.imag[gid], symbolic_iteration.placeholder[thread_id]);

				symbolic_iteration.size[gid] = std::distance(symbolic_iteration.placeholder[thread_id], end);

				/* compute hash */
				symbolic_iteration.hash[gid] = rule->hasher(symbolic_iteration.placeholder[thread_id], end);
			}
		}

		mid_step_function(3);
	}

	/*
	compute interferences
	*/
	void symbolic_iteration::compute_collisions() {
		num_object_after_interferences = num_object;
		bool fast = false;
		bool skip_test = num_object < utils::min_vector_size;
		size_t test_size = skip_test ? 0 : num_object*collision_test_proportion;

		/*
		function for partition
		*/
		auto static partitioner = [&](size_t const &gid) {
			/* check if graph is unique */
			if (!is_unique[gid])
				return false;

			/* check for zero probability */
			PROBA_TYPE r = real[gid];
			PROBA_TYPE i = imag[gid];

			return r*r + i*i > tolerance;
		};

		/*
		function for interferences
		*/
		auto static interferencer = [&](size_t const gid) {
			/* accessing key */
			tbb::concurrent_hash_map<size_t, size_t>::accessor it;
			if (elimination_map.insert(it, hash[gid])) {
				/* if it doesn't exist add it */
				it->second = gid;

				/* keep this graph */
				is_unique[gid] = true;
			} else {
				/* if it exist add the probabilities */
				real[it->second] += real[gid];
				imag[it->second] += imag[gid];

				/* discard this graph */
				is_unique[gid] = false;
			}
			it.release();
		};

		/* !!!!!!!!!!!!!!!!
		step (4)
		 !!!!!!!!!!!!!!!! */

		if (!skip_test) {
			#pragma omp parallel for schedule(static)
			for (size_t gid = 0; gid < test_size; ++gid) //size_t gid = gid[i];
				interferencer(gid);

			fast = test_size - elimination_map.size() < test_size*collision_test_proportion;

			/* check if we should continue */
			if (fast) {
				/* get all unique graphs with a non zero probability */
				auto partitioned_it = __gnu_parallel::partition(next_gid.begin(), next_gid.begin() + test_size, partitioner);
				partitioned_it = std::rotate(partitioned_it, next_gid.begin() + test_size, next_gid.begin() + num_object);
				num_object_after_interferences = std::distance(next_gid.begin(), partitioned_it);
			}
		}

		if (!fast)
			#pragma omp parallel for schedule(static)
			for (size_t gid = test_size; gid < num_object; ++gid) //size_t gid = gid[i];
				interferencer(gid);
				
		elimination_map.clear();

		auto partitioned_it = next_gid.begin() + num_object_after_interferences;
		if (!fast)
			/* get all unique graphs with a non zero probability */
			partitioned_it = __gnu_parallel::partition(next_gid.begin(), partitioned_it, partitioner);
		num_object_after_interferences = std::distance(next_gid.begin(), partitioned_it);
	}

	/*
	finalize iteration
	*/
	void symbolic_iteration::finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, debug_t mid_step_function=[](int){}) {
		long long int max_num_object;

		mid_step_function(4);

		/* !!!!!!!!!!!!!!!!
		step (5)
		 !!!!!!!!!!!!!!!! */

		max_num_object = (last_iteration.num_object + 
			next_iteration.num_object +
			get_additional_max_num_object(last_iteration, *this)) / 2;
		max_num_object = std::max(max_num_object, (long long int)utils::min_vector_size);

		if (num_object_after_interferences > max_num_object) {

			/* generate random selectors */
			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < num_object_after_interferences; ++i)  {
				size_t gid = next_gid[i];

				PROBA_TYPE r = real[gid];
				PROBA_TYPE i = imag[gid];

				double random_number = utils::unfiorm_from_hash(hash[gid]); //random_generator();
				random_selector[gid] = std::log( -std::log(1 - random_number) / (r*r + i*i));
			}

			/* select graphs according to random selectors */
			__gnu_parallel::nth_element(next_gid.begin(), next_gid.begin() + max_num_object, next_gid.begin() + num_object_after_interferences,
			[&](size_t const &gid1, size_t const &gid2) {
				return random_selector[gid1] < random_selector[gid2];
			});

			next_iteration.num_object = max_num_object;
		} else
			next_iteration.num_object = num_object_after_interferences;

		mid_step_function(5);

		/* !!!!!!!!!!!!!!!!
		step (6)
		 !!!!!!!!!!!!!!!! */

		/* sort to make memory access more continuous */
		__gnu_parallel::sort(next_gid.begin(), next_gid.begin() + next_iteration.num_object);

		/* resize new step variables */
		next_iteration.resize(next_iteration.num_object);

		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();
				
			/* prepare for partial sum */
			#pragma omp for schedule(static)
			for (size_t gid = 0; gid < next_iteration.num_object; ++gid) {
				size_t id = next_gid[gid];

				next_iteration.object_begin[gid + 1] = size[id];

				/* assign magnitude */
				next_iteration.real[gid] = real[id];
				next_iteration.imag[gid] = imag[id];
			}

			#pragma omp single
			{
				__gnu_parallel::partial_sum(next_iteration.object_begin.begin() + 1,
					next_iteration.object_begin.begin() + next_iteration.num_object + 1,
					next_iteration.object_begin.begin() + 1);

				next_iteration.allocate(next_iteration.object_begin[next_iteration.num_object]);

				mid_step_function(6);
			}

			/* !!!!!!!!!!!!!!!!
			step (7)
			 !!!!!!!!!!!!!!!! */

			PROBA_TYPE real_, imag_;

			#pragma omp for schedule(static)
			for (size_t gid = 0; gid < next_iteration.num_object; ++gid) {
				auto id = next_gid[gid];
				auto this_parent_gid = parent_gid[id];

				rule->populate_child(last_iteration.objects.begin() + last_iteration.object_begin[this_parent_gid],
					last_iteration.objects.begin() + last_iteration.object_begin[this_parent_gid + 1],
					child_id[id],
					real_, imag_,
					next_iteration.objects.begin() + next_iteration.object_begin[gid]);
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

		#pragma omp parallel for reduction(+:total_proba)
		for (size_t gid = 0; gid < num_object; ++gid) {
			PROBA_TYPE r = real[gid];
			PROBA_TYPE i = imag[gid];

			total_proba += r*r + i*i;
		}

		PROBA_TYPE normalization_factor = std::sqrt(total_proba);

		#pragma omp parallel for
		for (size_t gid = 0; gid < num_object; ++gid) {
			real[gid] /= normalization_factor;
			imag[gid] /= normalization_factor;
		}
	}
}