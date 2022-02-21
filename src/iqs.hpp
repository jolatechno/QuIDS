#pragma once

#include <parallel/algorithm>
#include <parallel/numeric>
#include <limits>

#include <complex>
#include <cstddef>
#include <vector>

#include "utils/libs/robin_hood.h"

#ifndef PROBA_TYPE
	#define PROBA_TYPE double
#endif
#ifndef TOLERANCE
	#define TOLERANCE 1e-30;
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
	#ifdef SIMPLE_TRUNCATION
		bool simple_truncation = true;
	#else
		bool simple_truncation = false;
	#endif

	/* forward typedef */
	typedef std::complex<PROBA_TYPE> mag_t;
	typedef class iteration it_t;
	typedef class symbolic_iteration sy_it_t;
	typedef class rule rule_t;
	typedef std::function<void(char* parent_begin, char* parent_end, mag_t &mag)> modifier_t;
	typedef std::function<void(const char* step)> debug_t;

	/* 
	rule virtual class
	*/
	class rule {
	public:
		rule() {};
		virtual inline void get_num_child(char const *parent_begin, char const *parent_end, size_t &num_child, size_t &max_child_size) const = 0;
		virtual inline void populate_child(char const *parent_begin, char const *parent_end, char* const child_begin, uint32_t const child_id, size_t &size, mag_t &mag) const = 0;
		virtual inline void populate_child_simple(char const *parent_begin, char const *parent_end, char* const child_begin, uint32_t const child_id) const { //can be overwritten
			size_t size_placeholder;
			mag_t mag_placeholder;
			populate_child(parent_begin, parent_end, child_begin, child_id,
				size_placeholder, mag_placeholder);
		}
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
		friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration, size_t max_num_object, debug_t mid_step_function);  
		friend void inline simulate(it_t &iteration, modifier_t const rule);

	protected:
		mutable size_t max_symbolic_object_size = 0;

		mutable std::vector<mag_t> magnitude;
		mutable std::vector<char> objects;
		mutable std::vector<size_t> object_begin;
		mutable std::vector<size_t> num_childs;

		void inline resize(size_t num_object) const {
			#pragma omp parallel sections
			{
				#pragma omp section
				utils::smart_resize(magnitude, num_object);

				#pragma omp section
				utils::smart_resize(num_childs, num_object + 1);

				#pragma omp section
				utils::smart_resize(object_begin, num_object + 1);
			}
		}
		void inline allocate(size_t size) const {
			utils::smart_resize(objects, size);
		}

		void compute_num_child(rule_t const *rule, debug_t mid_step_function=[](const char*){}) const;
		void generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function=[](const char*){}) const;
		void apply_modifier(modifier_t const rule);
		void normalize(debug_t mid_step_function=[](const char*){});

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
		size_t get_num_symbolic_object() const {
			return num_childs[num_object];
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
				#pragma omp for 
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
			object_begin_ = &objects[this_object_begin];
		}
		void get_object(size_t const object_id, char const *& object_begin_, size_t &object_size, mag_t &mag) const {
			size_t this_object_begin = object_begin[object_id];
			object_size = object_begin[object_id + 1] - this_object_begin;
			mag = magnitude[object_id];
			object_begin_ = &objects[this_object_begin];
		}
	};

	/*
	symboluc iteration class
	*/
	class symbolic_iteration {
		friend iteration;
		friend size_t inline get_max_num_object(it_t const &next_iteration, it_t const &last_iteration, sy_it_t const &symbolic_iteration);
		friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration, size_t max_num_object, debug_t mid_step_function); 

	protected:
		std::vector<robin_hood::unordered_map<size_t, size_t>> elimination_maps;
		std::vector<char*> placeholder;

		std::vector<mag_t> magnitude;
		std::vector<size_t> next_oid;
		std::vector<bool> is_unique;
		std::vector<size_t> size;
		std::vector<size_t> hash;
		std::vector<size_t> parent_oid;
		std::vector<uint32_t> child_id;
		std::vector<double> random_selector;
		std::vector<size_t> next_oid_partitioner_buffer;

		void inline resize(size_t num_object) {
			#pragma omp parallel sections
			{
				#pragma omp section
				utils::smart_resize(magnitude, num_object);

				#pragma omp section
				utils::smart_resize(next_oid, num_object);

				#pragma omp section
				utils::smart_resize(is_unique, num_object);

				#pragma omp section
				utils::smart_resize(size, num_object);

				#pragma omp section
				utils::smart_resize(hash, num_object);

				#pragma omp section
				utils::smart_resize(parent_oid, num_object);

				#pragma omp section
				utils::smart_resize(child_id, num_object);

				#pragma omp section
				utils::smart_resize(random_selector, num_object);

				#pragma omp section
				utils::smart_resize(next_oid_partitioner_buffer, num_object);
			}

			utils::parallel_iota(&next_oid[0], &next_oid[num_object], 0);
		}
		void inline reserve(size_t max_size) {
			int num_threads;
			#pragma omp parallel
			#pragma omp single
			num_threads = omp_get_num_threads();

			placeholder.resize(num_threads);

			#pragma omp parallel
			{
				auto &buffer = placeholder[omp_get_thread_num()];
				if (buffer == NULL)
					free(buffer);
				buffer = new char[max_size];
			}
		}

		void compute_collisions(debug_t mid_step_function=[](const char*){});
		void finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, const size_t max_num_object=0xffffffff, debug_t mid_step_function=[](const char*){});

	public:
		size_t num_object = 0;
		size_t num_object_after_interferences = 0;

		symbolic_iteration() {}
	};

	/*
	for memory managment
	*/
	size_t inline get_max_num_object(it_t const &next_iteration, it_t const &last_iteration, sy_it_t const &symbolic_iteration) {
		if (symbolic_iteration.num_object_after_interferences == 0)
			return -1;
		
		static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 2*sizeof(size_t);
		static const size_t symbolic_iteration_memory_size = 1 + 2*sizeof(PROBA_TYPE) + 7*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);

		// get the free memory and the total amount of memory...
		size_t free_mem;
		utils::get_free_mem(free_mem);

		// get each size
		size_t next_iteration_object_size = next_iteration.objects.capacity();
		size_t last_iteration_object_size = last_iteration.objects.capacity();

		size_t next_iteration_property_size = next_iteration.magnitude.capacity();
		size_t last_iteration_property_size = last_iteration.magnitude.capacity();

		size_t symbolic_iteration_size = symbolic_iteration.magnitude.capacity();

		size_t last_iteration_num_object = last_iteration.num_object;
		size_t symbolic_iteration_num_object = symbolic_iteration.num_object;

		// get the total memory
		size_t total_useable_memory = next_iteration_object_size + last_iteration_object_size + // size of objects
			(last_iteration_property_size + next_iteration_property_size)*iteration_memory_size + // size of properties
			symbolic_iteration_size*symbolic_iteration_memory_size + // size of symbolic properties
			free_mem; // free memory per shared memory simulation

		// compute average object size
		size_t iteration_size_per_object = 0;

		// compute the average size of an object for the next iteration:
		size_t test_size = std::max((size_t)1, (size_t)(size_average_proportion*symbolic_iteration.num_object_after_interferences));
		#pragma omp parallel for reduction(+:iteration_size_per_object)
		for (size_t oid = 0; oid < test_size; ++oid)
			iteration_size_per_object += symbolic_iteration.size[oid];
		iteration_size_per_object /= test_size;

		// add the cost of the symbolic iteration in itself
		iteration_size_per_object += symbolic_iteration_memory_size*symbolic_iteration_num_object/last_iteration_num_object/2; // size for symbolic iteration
		
		// and the constant size per object
		iteration_size_per_object += iteration_memory_size;

		// and the cost of unused space
		iteration_size_per_object *= utils::upsize_policy;

		return total_useable_memory / iteration_size_per_object * (1 - safety_margin);
	}

	/*
	simulation function
	*/
	void inline simulate(it_t &iteration, modifier_t const rule) {
		iteration.apply_modifier(rule);
	}
	void inline simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration, size_t max_num_object=0, debug_t mid_step_function=[](const char*){}) {
		iteration.compute_num_child(rule, mid_step_function);
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
		symbolic_iteration.compute_collisions(mid_step_function);

		if (max_num_object == 0) {
			mid_step_function("get_max_num_object");
			max_num_object = get_max_num_object(iteration_buffer, iteration, symbolic_iteration)/2;
		}

		symbolic_iteration.finalize(rule, iteration, iteration_buffer, max_num_object, mid_step_function);
		iteration_buffer.normalize(mid_step_function);

		std::swap(iteration_buffer, iteration);
	}

	/*
	compute num child
	*/
	void iteration::compute_num_child(rule_t const *rule, debug_t mid_step_function) const {
		mid_step_function("num_child");

		if (num_object == 0)
			return;

		/* !!!!!!!!!!!!!!!!
		step (1)
		 !!!!!!!!!!!!!!!! */

		max_symbolic_object_size = 0;

		#pragma omp parallel for  reduction(max:max_symbolic_object_size)
		for (size_t oid = 0; oid < num_object; ++oid) {
			size_t size;
			rule->get_num_child(&objects[object_begin[oid]],
				&objects[object_begin[oid + 1]],
				num_childs[oid + 1], size);
			max_symbolic_object_size = std::max(max_symbolic_object_size, size);
		}

		__gnu_parallel::partial_sum(&num_childs[1], &num_childs[num_object + 1], &num_childs[1]);
	}

	/*
	generate symbolic iteration
	*/
	void iteration::generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, debug_t mid_step_function) const {
		if (num_object == 0) {
			symbolic_iteration.num_object = 0;
			mid_step_function("prepare_index");
			mid_step_function("symbolic_iteration");
			return;
		}

		mid_step_function("prepare_index");

		/* !!!!!!!!!!!!!!!!
		step (2)
		 !!!!!!!!!!!!!!!! */

		symbolic_iteration.num_object = get_num_symbolic_object();

		/* resize symbolic iteration */
		symbolic_iteration.resize(symbolic_iteration.num_object);
		symbolic_iteration.reserve(max_symbolic_object_size);
		
		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();

			#pragma omp for 
			for (size_t oid = 0; oid < num_object; ++oid) {
				/* assign parent ids and child ids for each child */
				std::fill(&symbolic_iteration.parent_oid[num_childs[oid]],
					&symbolic_iteration.parent_oid[num_childs[oid + 1]],
					oid);
				std::iota(&symbolic_iteration.child_id[num_childs[oid]],
					&symbolic_iteration.child_id[num_childs[oid + 1]],
					0);
			}

			#pragma omp single
			mid_step_function("symbolic_iteration");

			/* !!!!!!!!!!!!!!!!
			step (3)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for 
			for (size_t oid = 0; oid < symbolic_iteration.num_object; ++oid) {
				auto id = symbolic_iteration.parent_oid[oid];

				/* generate graph */
				symbolic_iteration.magnitude[oid] = magnitude[id];
				rule->populate_child(&objects[object_begin[id]],
					&objects[object_begin[id + 1]],
					symbolic_iteration.placeholder[thread_id], symbolic_iteration.child_id[oid],
					symbolic_iteration.size[oid], symbolic_iteration.magnitude[oid]);

				/* compute hash */
				symbolic_iteration.hash[oid] = rule->hasher(symbolic_iteration.placeholder[thread_id],
					symbolic_iteration.placeholder[thread_id] + symbolic_iteration.size[oid]);
			}
		}
	}

	/*
	compute interferences
	*/
	void symbolic_iteration::compute_collisions(debug_t mid_step_function) {
		if (num_object == 0) {
			num_object_after_interferences = 0;
			mid_step_function("compute_collisions - prepare");
			mid_step_function("compute_collisions - insert");
			mid_step_function("compute_collisions - finalize");
			return;
		}

		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();
		
		elimination_maps.resize(num_threads);

		int const num_bucket = utils::nearest_power_of_two(load_balancing_bucket_per_thread*num_threads);
		size_t const offset = 8*sizeof(size_t) - utils::log_2_upper_bound(num_bucket);

		const auto compute_interferences = [&](size_t *end_iterator, bool first) {
			mid_step_function("compute_collisions - prepare");

			size_t oid_end = std::distance(&next_oid[0], end_iterator);

			std::vector<size_t> modulo_offset(num_threads*(num_threads + 1) + 1, 0);

			
			std::vector<int> load_balancing_begin(num_threads + 1, 0);
			std::vector<size_t> partition_begin(num_threads*(num_bucket + 1), 0);
			std::vector<size_t> total_partition_begin(num_bucket + 1, 0);

			std::vector<size_t> load_begin(num_threads + 1, 0);
			load_begin[0] = 0;

			const auto partitioner = [&](size_t const oid) {
				return hash[oid] >> offset;
			};
			
			/* partition to limit collisions */
			#pragma omp parallel
			{
				int thread_id = omp_get_thread_num();
				auto &elimination_map = elimination_maps[thread_id];

				load_begin[thread_id + 1] = (thread_id + 1) * oid_end / num_threads;

				#pragma omp barrier

				size_t this_oid_begin = load_begin[thread_id];
				size_t this_oid_end = load_begin[thread_id + 1];

				if (first) {
					utils::generalized_partition_from_iota(&next_oid[this_oid_begin], &next_oid[this_oid_end], this_oid_begin,
						&partition_begin[(num_bucket + 1)*thread_id], &partition_begin[(num_bucket + 1)*(thread_id + 1)],
						partitioner);
				} else
					utils::generalized_partition(&next_oid[this_oid_begin], &next_oid[this_oid_end], &next_oid_partitioner_buffer[this_oid_begin],
						&partition_begin[(num_bucket + 1)*thread_id], &partition_begin[(num_bucket + 1)*(thread_id + 1)],
						partitioner);

				/* compute total partition for load balancing */
				for (int i = 1; i <= num_bucket; ++i)
					#pragma omp atomic
					total_partition_begin[i] += partition_begin[(num_bucket + 1)*thread_id + i];

				/* compute load balancing */
				#pragma omp barrier
				#pragma omp single 
					utils::load_balancing_from_prefix_sum(total_partition_begin.begin(), total_partition_begin.end(),
						load_balancing_begin.begin(), load_balancing_begin.end());

				if (thread_id == 0)
					mid_step_function("compute_collisions - insert");

				/* insert into separate hashmaps */
				for (int j = load_balancing_begin[thread_id]; j < load_balancing_begin[thread_id + 1]; ++j) {
					size_t total_size = 0;
					for (int other_thread_id = 0; other_thread_id < num_threads; ++other_thread_id) {
						size_t begin = partition_begin[(num_bucket + 1)*other_thread_id + j];
						size_t end = partition_begin[(num_bucket + 1)*other_thread_id + j + 1];

						total_size += end - begin;
					}

					elimination_map.reserve(total_size);

					for (int other_thread_id = 0; other_thread_id < num_threads; ++other_thread_id) {
						size_t other_oid_begin = load_begin[other_thread_id];

						size_t begin = partition_begin[(num_bucket + 1)*other_thread_id + j] + other_oid_begin;
						size_t end = partition_begin[(num_bucket + 1)*other_thread_id + j + 1] + other_oid_begin;

						for (size_t i = begin; i < end; ++i) {
							size_t oid = next_oid[i];

							/* accessing key */
							auto [it, unique] = elimination_map.insert({hash[oid], oid});
							if (!unique)
								/* if it exist add the probabilities */
								magnitude[it->second] += magnitude[oid];
							/* keep the object if it is unique */
							is_unique[oid] = unique;
						}
					}

					elimination_map.clear();
				}
			}

			mid_step_function("compute_collisions - finalize");

			/* check for zero probability */
			return __gnu_parallel::partition(&next_oid[0], end_iterator,
				[&](size_t const &oid) {
					if (!is_unique[oid])
						return false;

					return std::norm(magnitude[oid]) > tolerance;
				});
		};

		/* !!!!!!!!!!!!!!!!
		step (4)
		 !!!!!!!!!!!!!!!! */

		bool fast = false;
		bool skip_test = collision_test_proportion == 0 || collision_tolerance == 0 || num_object < min_collision_size;
		size_t test_size = num_object*collision_test_proportion;
		size_t *partitioned_it = &next_oid[0] + (skip_test ? num_object : test_size);

		/* get all unique graphs with a non zero probability */
		if (!skip_test) {
			auto test_partitioned_it = compute_interferences(partitioned_it, true);

			size_t number_of_collision = std::distance(test_partitioned_it, partitioned_it);
			fast = number_of_collision < test_size*collision_tolerance;

			partitioned_it = std::rotate(test_partitioned_it, partitioned_it, &next_oid[num_object]);
		}
		if (!fast)
			partitioned_it = compute_interferences(partitioned_it, skip_test);
			
		num_object_after_interferences = std::distance(&next_oid[0], partitioned_it);
	}

	/*
	finalize iteration
	*/
	void symbolic_iteration::finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, const size_t max_num_object, debug_t mid_step_function) {
		if (num_object_after_interferences == 0) {
			next_iteration.num_object = 0;
			mid_step_function("truncate");
			mid_step_function("prepare_final");
			mid_step_function("final");
			return;
		}
		
		mid_step_function("truncate");

		/* !!!!!!!!!!!!!!!!
		step (5)
		 !!!!!!!!!!!!!!!! */

		if (max_num_object > utils::min_vector_size && num_object_after_interferences > max_num_object) {
			if (simple_truncation) {
				/* select graphs according to random selectors */
				utils::parallel_aprox_nth_element(&next_oid[0], &next_oid[max_num_object], &next_oid[num_object_after_interferences],
				[&](size_t const &oid1, size_t const &oid2) {
					return std::norm(magnitude[oid1]) < std::norm(magnitude[oid2]);
				});

			} else {
				/* generate random selectors */
				#pragma omp parallel for 
				for (size_t i = 0; i < num_object_after_interferences; ++i)  {
					size_t oid = next_oid[i];

					double random_number = utils::unfiorm_from_hash(hash[oid]); //random_generator();
					random_selector[oid] = std::log( -std::log(1 - random_number) / std::norm(magnitude[oid]));
				}

				/* select graphs according to random selectors */
				utils::parallel_aprox_nth_element(&next_oid[0], &next_oid[max_num_object], &next_oid[num_object_after_interferences],
				[&](size_t const &oid1, size_t const &oid2) {
					return random_selector[oid1] < random_selector[oid2];
				});
			}

			next_iteration.num_object = max_num_object;
		} else
			next_iteration.num_object = num_object_after_interferences;

		mid_step_function("prepare_final");

		/* !!!!!!!!!!!!!!!!
		step (6)
		 !!!!!!!!!!!!!!!! */

		/* sort to make memory access more continuous */
		__gnu_parallel::sort(&next_oid[0], &next_oid[next_iteration.num_object]);

		/* resize new step variables */
		next_iteration.resize(next_iteration.num_object);
				
		/* prepare for partial sum */
		#pragma omp parallel for 
		for (size_t oid = 0; oid < next_iteration.num_object; ++oid) {
			auto id = next_oid[oid];

			/* assign magnitude and size */
			next_iteration.object_begin[oid + 1] = size[id];
			next_iteration.magnitude[oid] = magnitude[id];
		}

		__gnu_parallel::partial_sum(&next_iteration.object_begin[1],
			&next_iteration.object_begin[next_iteration.num_object + 1],
			&next_iteration.object_begin[1]);

		next_iteration.allocate(next_iteration.object_begin[next_iteration.num_object]);

		mid_step_function("final");

		/* !!!!!!!!!!!!!!!!
		step (7)
		 !!!!!!!!!!!!!!!! */

		#pragma omp parallel for 
		for (size_t oid = 0; oid < next_iteration.num_object; ++oid) {
			auto id = next_oid[oid];
			auto this_parent_oid = parent_oid[id];
				
			rule->populate_child_simple(&last_iteration.objects[last_iteration.object_begin[this_parent_oid]],
				&last_iteration.objects[last_iteration.object_begin[this_parent_oid + 1]],
				&next_iteration.objects[next_iteration.object_begin[oid]],
				child_id[id]);
		}
	}

	/*
	apply modifier
	*/
	void iteration::apply_modifier(modifier_t const rule) {
		#pragma omp parallel for 
		for (size_t oid = 0; oid < num_object; ++oid)
			/* generate graph */
			rule(&objects[object_begin[oid]],
				&objects[object_begin[oid + 1]],
				magnitude[oid]);
	}

	/*
	normalize
	*/
	void iteration::normalize(debug_t mid_step_function) {
		total_proba = 0;

		if (num_object == 0) {
			mid_step_function("normalize");
			mid_step_function("end");
			return;
		}

		mid_step_function("normalize");

		/* !!!!!!!!!!!!!!!!
		step (8)
		 !!!!!!!!!!!!!!!! */

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
		
		mid_step_function("end");
	}
}