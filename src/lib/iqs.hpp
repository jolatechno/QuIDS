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
#ifndef COLLISION_PROBABILITY_TOLERANCE
	#define COLLISION_PROBABILITY_TOLERANCE 0.15
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
		#include "utils/load_balancing.hpp"
		#include "utils/memory.hpp"
		#include "utils/random.hpp"
		#include "utils/vector.hpp"
	}

	#include "utils/libs/boost_hash.hpp"
	
	/*
	global variable definition
	*/
	PROBA_TYPE tolerance = TOLERANCE;
	float safety_margin = SAFETY_MARGIN;
	float collision_test_proportion = COLLISION_TEST_PROPORTION;
	float collision_tolerance = COLLISION_TOLERANCE;
	float collision_probability_tolerance = COLLISION_PROBABILITY_TOLERANCE;

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

	/* 
	rule virtual class
	*/
	class rule {
	public:
		rule() {};
		virtual inline void get_num_child(char* object_begin, char* object_end, uint16_t &num_child, size_t &max_child_size) const = 0;
		virtual inline char* populate_child(char* parent_begin, char* parent_end, uint16_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const = 0;
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
		mutable utils::numa_vector<uint16_t> num_childs;

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

			for (auto i = 0; i < size; ++i)
				objects[offset + i] = object_begin_[i];

			real[num_object - 1] = real_; imag[num_object - 1] = imag_;
			object_begin[num_object] = offset + size;
		}
		void inline generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration) const;
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
		utils::numa_vector<uint16_t> child_id;
		utils::numa_vector<bool> is_unique;
		utils::numa_vector<double> random_selector;

		void inline resize(size_t num_object) {
			size.zero_resize(num_object);
			hash.zero_resize(num_object);
			real.resize(num_object);
			imag.resize(num_object);
			next_gid.iota_resize(num_object);
			parent_gid.resize(num_object);
			child_id.resize(num_object);
			random_selector.zero_resize(num_object);
		}
		void inline reserve(size_t max_size) {
			#pragma omp parallel
			{
				auto &buffer = placeholder[omp_get_thread_num()];
				if (buffer == NULL)
					free(buffer);
				buffer = new char[max_size];
				for (auto i = 0; i < max_size; ++i) buffer[i] = 0; // touch
			}
		}

	public:
		symbolic_iteration() {}
		void inline finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration);
	};

	/*
	for memory managment
	*/
	long long int inline get_additional_max_num_object(it_t const &last_iteration, sy_it_t const &symbolic_iteration) {
		// get the free memory and the total amount of memory...
		auto [total_memory, free_mem] = utils::get_mem_usage_and_free_mem();

		// and according to the "safety_margin" (a proportion of total memory) compute the total delta between the amount free memory and the target
		long int mem_difference = free_mem - total_memory*safety_margin;

		static long long int iteration_size = 2*sizeof(PROBA_TYPE) + sizeof(size_t) + sizeof(uint16_t);
		static long long int symbolic_iteration_size = 1 + 2*sizeof(PROBA_TYPE) + 6*sizeof(size_t) + sizeof(uint16_t) + sizeof(double);

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
	void simulate(it_t &iteration, rule_t const *rule, it_t &iteration_buffer, sy_it_t &symbolic_iteration) {
		iteration.generate_symbolic_iteration(rule, symbolic_iteration);
		symbolic_iteration.finalize(rule, iteration, iteration_buffer);
		std::swap(iteration_buffer, iteration);
	}

	/*
	generate symbolic iteration
	*/
	void inline iteration::generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration) const {
		size_t max_size = 0;
		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();

			/* !!!!!!!!!!!!!!!!
			step (1)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for schedule(static) reduction(max:max_size)
			for (auto gid = 0; gid < num_object; ++gid) {
				size_t size;
				rule->get_num_child(objects.begin() + object_begin[gid],
					objects.begin() + object_begin[gid + 1],
					num_childs[gid + 1], size);
				max_size = std::max(max_size, size);
			}

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
			for (auto gid = 0; gid < num_object; ++gid) {
				/* assign parent ids and child ids for each child */
				std::fill(symbolic_iteration.parent_gid.begin() + num_childs[gid],
					symbolic_iteration.parent_gid.begin() + num_childs[gid + 1],
					gid);
				std::iota(symbolic_iteration.child_id.begin() + num_childs[gid],
					symbolic_iteration.child_id.begin() + num_childs[gid + 1],
					0);
			}
		}
	}

	/*
	finalize iteration
	*/
	void inline symbolic_iteration::finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration) {
		double &total_proba = next_iteration.total_proba;
		total_proba = 0;

		num_object_after_interferences = num_object;

		bool fast = false;
		bool skip_test = num_object < utils::min_vector_size || last_iteration.total_proba - 1 > collision_probability_tolerance;
		size_t test_size = skip_test ? 0 : num_object*collision_test_proportion;
		long long int max_num_object;

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

		/*
		actual code
		*/
		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();

			/* !!!!!!!!!!!!!!!!
			step (3)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for schedule(static)
			for (auto gid = 0; gid < num_object; ++gid) {
				auto id = parent_gid[gid];

				/* generate graph */
				real[gid] = last_iteration.real[id];
				imag[gid] = last_iteration.imag[id];
				char* end = rule->populate_child(last_iteration.objects.begin() + last_iteration.object_begin[id],
					last_iteration.objects.begin() + last_iteration.object_begin[id + 1],
					child_id[gid],
					real[gid], imag[gid], placeholder[thread_id]);

				size[gid] = std::distance(placeholder[thread_id], end);

				/* compute hash */
				hash[gid] = 0;
				for (char* it = placeholder[thread_id]; it != end; ++it)
					boost::hash_combine(hash[gid], *it);
			}

			/* !!!!!!!!!!!!!!!!
			step (4)
			 !!!!!!!!!!!!!!!! */

			if (!skip_test) {
				#pragma omp for schedule(static)
				for (auto gid = 0; gid < test_size; ++gid) //size_t gid = gid[i];
					interferencer(gid);

				#pragma omp barrier

				#pragma omp single
				{
					fast = test_size - elimination_map.size() < test_size*collision_test_proportion;

					/* check if we should continue */
					if (fast) {
						/* get all unique graphs with a non zero probability */
						auto partitioned_it = __gnu_parallel::partition(next_gid.begin(), next_gid.begin() + test_size, partitioner);
						partitioned_it = std::rotate(partitioned_it, next_gid.begin() + test_size, next_gid.begin() + num_object);
						num_object_after_interferences = std::distance(next_gid.begin(), partitioned_it);
					}
				}
			}

			#pragma omp barrier

			if (!fast)
				#pragma omp for schedule(static)
				for (auto gid = test_size; gid < num_object; ++gid) //size_t gid = gid[i];
					interferencer(gid);

			#pragma omp single
			{
				elimination_map.clear();

				auto partitioned_it = next_gid.begin() + num_object_after_interferences;
				if (!fast)
					/* get all unique graphs with a non zero probability */
					partitioned_it = __gnu_parallel::partition(next_gid.begin(), partitioned_it, partitioner);
				num_object_after_interferences = std::distance(next_gid.begin(), partitioned_it);

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
					for (auto gid_it = next_gid.begin(); gid_it != partitioned_it; ++gid_it)  {
						PROBA_TYPE r = real[*gid_it];
						PROBA_TYPE i = imag[*gid_it];

						double random_number = utils::unfiorm_from_hash(hash[*gid_it]); //random_generator();
						random_selector[*gid_it] = std::log( -std::log(1 - random_number) / (r*r + i*i));
					}

					/* select graphs according to random selectors */
					__gnu_parallel::nth_element(next_gid.begin(), next_gid.begin() + max_num_object, partitioned_it,
					[&](size_t const &gid1, size_t const &gid2) {
						return random_selector[gid1] < random_selector[gid2];
					});

					next_iteration.num_object = max_num_object;
				} else
					next_iteration.num_object = num_object_after_interferences;

				/* !!!!!!!!!!!!!!!!
				step (6)
				 !!!!!!!!!!!!!!!! */

				/* sort to make memory access more continuous */
				__gnu_parallel::sort(next_gid.begin(), next_gid.begin() + next_iteration.num_object);

				/* resize new step variables */
				next_iteration.resize(next_iteration.num_object);
			}
				
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
			}

			/* !!!!!!!!!!!!!!!!
			step (7)
			 !!!!!!!!!!!!!!!! */

			PROBA_TYPE real_, imag_;

			#pragma omp for schedule(static)
			for (auto gid = 0; gid < next_iteration.num_object; ++gid) {
				auto id = next_gid[gid];
				auto this_parent_gid = parent_gid[id];

				rule->populate_child(last_iteration.objects.begin() + last_iteration.object_begin[this_parent_gid],
					last_iteration.objects.begin() + last_iteration.object_begin[this_parent_gid + 1],
					child_id[id],
					real_, imag_,
					next_iteration.objects.begin() + next_iteration.object_begin[gid]);
			}

			/* !!!!!!!!!!!!!!!!
			step (8)
			 !!!!!!!!!!!!!!!! */

			#pragma omp for reduction(+:total_proba)
			for (auto gid = 0; gid < next_iteration.num_object; ++gid) {
				PROBA_TYPE r = next_iteration.real[gid];
				PROBA_TYPE i = next_iteration.imag[gid];

				total_proba += r*r + i*i;
			}

			#pragma omp single
			total_proba = std::sqrt(total_proba);

			#pragma omp for
			for (auto gid = 0; gid < next_iteration.num_object; ++gid) {
				next_iteration.real[gid] /= total_proba;
				next_iteration.imag[gid] /= total_proba;
			}
		}

		total_proba *= total_proba;
	}
}