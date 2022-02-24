#pragma once

#include <parallel/algorithm>
#include <parallel/numeric>
#include <limits>



#include <iostream>




#include <complex>
#include <cstddef>
#include <vector>

#include "utils/libs/robin_hood.h"

#ifndef PROBA_TYPE
	#define PROBA_TYPE double
#endif
#ifndef HASH_MAP_OVERHEAD
	#define HASH_MAP_OVERHEAD 1.7
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
#ifndef LOAD_BALANCING_BUCKET_PER_THREAD
	#define LOAD_BALANCING_BUCKET_PER_THREAD 32
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
		#include "utils/vector.hpp"
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
		friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &next_iteration, sy_it_t &symbolic_iteration, size_t max_num_object, debug_t mid_step_function);  
		friend void inline simulate(it_t &iteration, modifier_t const rule);

	protected:
		mutable size_t max_symbolic_object_size = 0;

		mutable utils::fast_vector<mag_t> magnitude;
		mutable utils::fast_vector<char> objects;
		mutable utils::fast_vector<size_t> object_begin;
		mutable utils::fast_vector<size_t> num_childs;
		mutable utils::fast_vector<size_t> child_begin;
		mutable utils::fast_vector<size_t> truncated_oid;

		void inline resize(size_t num_object) const {
			#pragma omp parallel sections
			{
				#pragma omp section
				magnitude.resize(num_object);

				#pragma omp section
				num_childs.resize(num_object);

				#pragma omp section
				object_begin.resize(num_object + 1);

				#pragma omp section
				child_begin.resize(num_object + 1);

				#pragma omp section
				{
					truncated_oid.resize(num_object);
					utils::parallel_iota(&truncated_oid[0], &truncated_oid[0] + num_object, 0);
				}
			}
		}
		void inline allocate(size_t size) const {
			objects.resize(size);
		}


		/*
		utils functions
		*/
		size_t get_mem_size() const {
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t);
			return magnitude.size()*iteration_memory_size + objects.size();
		}
		float get_average_num_child() const {
			return (float)get_num_symbolic_object() / (float)num_object;
		}
		size_t get_object_length() const {
			return object_begin[num_object];
		}
		float get_average_object_size() const {
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t);
			return (float)iteration_memory_size + (float)get_object_length()/(float)num_object;
		}



		void compute_num_child(rule_t const *rule, debug_t mid_step_function=[](const char*){}) const;
		void generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, it_t const &next_iteration, const size_t max_num_object, debug_t mid_step_function=[](const char*){}) const;
		void apply_modifier(modifier_t const rule);
		void normalize(debug_t mid_step_function=[](const char*){});

	public:
		size_t num_object = 0;
		PROBA_TYPE total_proba = 1;

		iteration() {
			resize(0);
			allocate(0);
			object_begin[0] = 0;
		}
		iteration(char* object_begin_, char* object_end_) : iteration() {
			append(object_begin_, object_end_);
		}
		size_t get_num_symbolic_object() const {
			return __gnu_parallel::accumulate(&num_childs[0], &num_childs[0] + num_object, 0);
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
		friend void inline simulate(it_t &iteration, rule_t const *rule, it_t &next_iteration, sy_it_t &symbolic_iteration, size_t max_num_object, debug_t mid_step_function); 

	protected:
		std::vector<robin_hood::unordered_map<size_t, size_t>> elimination_maps;
		std::vector<char*> placeholder;

		utils::fast_vector<mag_t> magnitude;
		utils::fast_vector<size_t> next_oid;
		utils::fast_vector<char /*bool*/> is_unique;
		utils::fast_vector<size_t> size;
		utils::fast_vector<size_t> hash;
		utils::fast_vector<size_t> parent_oid;
		utils::fast_vector<uint32_t> child_id;
		utils::fast_vector<double> random_selector;
		utils::fast_vector<size_t> next_oid_partitioner_buffer;

		void inline resize(size_t num_object) {
			#pragma omp parallel sections
			{
				#pragma omp section
				magnitude.resize(num_object);

				#pragma omp section
				next_oid.resize(num_object);

				#pragma omp section
				is_unique.resize(num_object);

				#pragma omp section
				size.resize(num_object);

				#pragma omp section
				hash.resize(num_object);

				#pragma omp section
				parent_oid.resize(num_object);

				#pragma omp section
				child_id.resize(num_object);

				#pragma omp section
				random_selector.resize(num_object);

				#pragma omp section
				next_oid_partitioner_buffer.resize(num_object);
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


		/*
		utils functions
		*/
		float get_average_object_size() {
			static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);
			static const size_t symbolic_iteration_memory_size = 1 + 2*sizeof(PROBA_TYPE) + 5*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);
			return (float)hash_map_size + hash_map_size;
		}
		size_t get_mem_size() {
			static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);

			size_t memory_size = 0;
			for (auto &map : elimination_maps)
				memory_size += elimination_maps.capacity();
			memory_size *= hash_map_size;

			static const size_t symbolic_iteration_memory_size = 1 + 2*sizeof(PROBA_TYPE) + 5*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);
			memory_size += magnitude.size()*symbolic_iteration_memory_size;

			return memory_size;
		}
		float get_average_child_size() const {
			float average_size = (float)__gnu_parallel::accumulate(size.begin(), size.begin() + num_object, 0) / (float)num_object;

			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t);
			return (float)iteration_memory_size + average_size;
		}

		void compute_collisions(debug_t mid_step_function=[](const char*){});
		void finalize(rule_t const *rule, it_t const &last_iteration, it_t &next_iteration, const size_t max_num_object=0xffffffff, debug_t mid_step_function=[](const char*){});

	public:
		size_t num_object = 0;
		size_t num_object_after_interferences = 0;

		symbolic_iteration() {}
	};

	/*
	simulation function
	*/
	void inline simulate(it_t &iteration, modifier_t const rule) {
		iteration.apply_modifier(rule);
	}
	void inline simulate(it_t &iteration, rule_t const *rule, it_t &next_iteration, sy_it_t &symbolic_iteration, size_t max_num_object=0, debug_t mid_step_function=[](const char*){}) {
		/* compute the number of child */
		iteration.compute_num_child(rule, mid_step_function);

		/* max_num_object */
		if (max_num_object == 0) {
			mid_step_function("get_max_num_object");
			max_num_object = (float)(utils::get_free_mem() + next_iteration.get_mem_size() + symbolic_iteration.get_mem_size()) /
				(iteration.get_average_object_size() + iteration.get_average_num_child()*symbolic_iteration.get_average_object_size()) *
				(1 - safety_margin)/utils::upsize_policy;
		}

		/* generate symbolic iteration */
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, next_iteration, max_num_object, mid_step_function);
		symbolic_iteration.compute_collisions(mid_step_function);

		/* second max_num_object */
		if (max_num_object == 0) {
			mid_step_function("get_max_num_object");
			max_num_object = (float)(utils::get_free_mem() + next_iteration.get_mem_size()) /
				symbolic_iteration.get_average_child_size() *
				(1 - safety_margin)/utils::upsize_policy;
		}

		/* finish simulation */
		symbolic_iteration.finalize(rule, iteration, next_iteration, max_num_object, mid_step_function);
		next_iteration.normalize(mid_step_function);
	}

	/*
	compute num child
	*/
	void iteration::compute_num_child(rule_t const *rule, debug_t mid_step_function) const {
		/* !!!!!!!!!!!!!!!!
		num_child
		 !!!!!!!!!!!!!!!! */
		mid_step_function("num_child");

		if (num_object == 0)
			return;

		max_symbolic_object_size = 0;

		#pragma omp parallel for  reduction(max:max_symbolic_object_size)
		for (size_t oid = 0; oid < num_object; ++oid) {
			size_t size;
			rule->get_num_child(&objects[object_begin[oid]],
				&objects[object_begin[oid + 1]],
				num_childs[oid], size);
			max_symbolic_object_size = std::max(max_symbolic_object_size, size);
		}
	}

	/*
	generate symbolic iteration
	*/
	void iteration::generate_symbolic_iteration(rule_t const *rule, sy_it_t &symbolic_iteration, it_t const &next_iteration, const size_t max_num_object, debug_t mid_step_function) const {
		if (num_object == 0) {
			symbolic_iteration.num_object = 0;
			mid_step_function("truncate");
			mid_step_function("prepare_index");
			mid_step_function("symbolic_iteration");
			return;
		}

		

		/* !!!!!!!!!!!!!!!!
		pre_truncate
		 !!!!!!!!!!!!!!!! */
		mid_step_function("truncate");

		size_t truncated_num_object = num_object;
		if (max_num_object > utils::min_vector_size && num_object > max_num_object) {
			//if (simple_truncation) {
				/* select graphs according to random selectors */
				__gnu_parallel::nth_element(&truncated_oid[0], &truncated_oid[max_num_object], &truncated_oid[num_object],
				[&](size_t const &oid1, size_t const &oid2) {
					return std::norm(magnitude[oid1]) < std::norm(magnitude[oid2]);
				});

			/* !!!!!!
			TODO
			!!!!!! */
			/*} else {
				if (symbolic_iteration.num_object < num_object)
					symbolic_iteration.resize(num_object);

				/* generate random selectors */
				/*#pragma omp parallel for 
				for (size_t i = 0; i < num_object; ++i)  {
					size_t oid = next_oid[i];

					double random_number = utils::unfiorm_from_hash(hash[oid]); //random_generator();
					symbolic_iteration.random_selector[oid] = std::log( -std::log(1 - random_number) / std::norm(magnitude[oid]));
				}

				/* select graphs according to random selectors */
				/*__gnu_parallel::nth_element(&next_oid[0], &next_oid[max_num_object], &next_oid[num_object_after_interferences],
				[&](size_t const &oid1, size_t const &oid2) {
					return random_selector[oid1] < random_selector[oid2];
				});
			}*/

			truncated_num_object = max_num_object;
		}

		/* downsize if needed */
		size_t next_object_size = truncated_num_object*get_object_length()/num_object;
		if (next_object_size < next_iteration.get_object_length())
			next_iteration.allocate(next_object_size);

		/* downsize if needed */
		if (truncated_num_object < next_iteration.num_object)
			next_iteration.resize(truncated_num_object);

		





		/* !!!!!!!!!!!!!!!!
		prepare_index
		 !!!!!!!!!!!!!!!! */
		mid_step_function("prepare_index");

		child_begin[0] = 0;
		for (size_t i = 0; i < truncated_num_object; ++i) {
			size_t oid = truncated_oid[i];

			child_begin[i + 1] = child_begin[i] + num_childs[oid];
		}
		symbolic_iteration.num_object = child_begin[truncated_num_object];

		/* resize symbolic iteration */
		symbolic_iteration.resize(symbolic_iteration.num_object);
		symbolic_iteration.reserve(max_symbolic_object_size);
		
		#pragma omp parallel
		{
			auto thread_id = omp_get_thread_num();

			#pragma omp for 
			for (size_t i = 0; i < truncated_num_object; ++i) {
				size_t oid = truncated_oid[i];

				/* assign parent ids and child ids for each child */
				std::fill(&symbolic_iteration.parent_oid[child_begin[i]],
					&symbolic_iteration.parent_oid[child_begin[i + 1]],
					oid);
				std::iota(&symbolic_iteration.child_id[child_begin[i]],
					&symbolic_iteration.child_id[child_begin[i + 1]],
					0);
			}




			/* !!!!!!!!!!!!!!!!
			symbolic_iteration
			 !!!!!!!!!!!!!!!! */
			#pragma omp single
			mid_step_function("symbolic_iteration");

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

		std::vector<int> load_balancing_begin(num_threads + 1, 0);
		std::vector<size_t> partition_begin(num_bucket + 1, 0);





		/* !!!!!!!!!!!!!!!!
		partition
		!!!!!!!!!!!!!!!! */
		iqs::utils::parallel_generalized_partition_from_iota(&next_oid[0], &next_oid[0] + num_object, 0,
			&partition_begin[0], &partition_begin[num_bucket + 1],
			[&](size_t const oid) {
				return hash[oid] >> offset;
			});





		/* !!!!!!!!!!!!!!!!
		load-balance
		!!!!!!!!!!!!!!!! */
		utils::load_balancing_from_prefix_sum(partition_begin.begin(), partition_begin.end(),
			load_balancing_begin.begin(), load_balancing_begin.end());






		/* !!!!!!!!!!!!!!!!
		compute-collision
		!!!!!!!!!!!!!!!! */
		mid_step_function("compute_collisions - insert");
		#pragma omp parallel
		{
			int thread_id = omp_get_thread_num();
			auto &elimination_map = elimination_maps[thread_id];

			int load_begin = load_balancing_begin[thread_id], load_end = load_balancing_begin[thread_id + 1];
			for (int j = load_begin; j < load_end; ++j) {
				size_t begin = partition_begin[j], end = partition_begin[j + 1];

				elimination_map.reserve(end - begin);
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
				elimination_map.clear();
			}
		}
		mid_step_function("compute_collisions - finalize");





		/* !!!!!!!!!!!!!!!!
		partition
		!!!!!!!!!!!!!!!! */
		size_t* partitioned_it = __gnu_parallel::partition(&next_oid[0], &next_oid[0] + num_object,
			[&](size_t const &oid) {
				if (!is_unique[oid])
					return false;

				return std::norm(magnitude[oid]) > tolerance;
			});
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
		



		/* !!!!!!!!!!!!!!!!
		truncate
		 !!!!!!!!!!!!!!!! */
		mid_step_function("truncate");

		if (max_num_object > utils::min_vector_size && num_object_after_interferences > max_num_object) {
			if (simple_truncation) {
				/* select graphs according to random selectors */
				__gnu_parallel::nth_element(&next_oid[0], &next_oid[max_num_object], &next_oid[num_object_after_interferences],
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
				__gnu_parallel::nth_element(&next_oid[0], &next_oid[max_num_object], &next_oid[num_object_after_interferences],
				[&](size_t const &oid1, size_t const &oid2) {
					return random_selector[oid1] < random_selector[oid2];
				});
			}

			next_iteration.num_object = max_num_object;
		} else
			next_iteration.num_object = num_object_after_interferences;





		/* !!!!!!!!!!!!!!!!
		prepare_final
		 !!!!!!!!!!!!!!!! */
		mid_step_function("prepare_final");

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

		


		/* !!!!!!!!!!!!!!!!
		final
		 !!!!!!!!!!!!!!!! */
		mid_step_function("final");

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




		/* !!!!!!!!!!!!!!!!
		normalize
		 !!!!!!!!!!!!!!!! */
		mid_step_function("normalize");

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