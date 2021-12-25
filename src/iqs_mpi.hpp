#pragma once

#include "iqs.hpp"

#include <mpi.h>

#ifndef MIN_EQUALIZE_SIZE
	#define MIN_EQUALIZE_SIZE 100
#endif
#ifndef EQUALIZE_IMBALANCE
	#define EQUALIZE_IMBALANCE 0.01
#endif

namespace iqs::mpi {
	namespace utils {        
		#include "utils/mpi_utils.hpp"
	}

	/* mpi auto type */
	const static MPI_Datatype Proba_MPI_Datatype = utils::get_mpi_datatype((PROBA_TYPE)0);
	const static MPI_Datatype mag_MPI_Datatype = utils::get_mpi_datatype((std::complex<PROBA_TYPE>)0);

	/* 
	global variables
	*/
	size_t min_equalize_size = MIN_EQUALIZE_SIZE;
	float equalize_imablance = EQUALIZE_IMBALANCE;

	/* forward typedef */
	typedef class mpi_iteration mpi_it_t;
	typedef class mpi_symbolic_iteration mpi_sy_it_t;

	/*
	mpi iteration class
	*/
	class mpi_iteration : public iqs::iteration {
		friend mpi_symbolic_iteration;
		friend size_t inline get_max_num_object(mpi_it_t const &next_iteration, mpi_it_t const &last_iteration, mpi_sy_it_t const &symbolic_iteration, MPI_Comm communicator);
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, iqs::debug_t mid_step_function);

	protected:
		void equalize_symbolic(MPI_Comm communicator);
		void normalize(MPI_Comm communicator, std::function<void()> mid_step_function=[](){});

	public:
		PROBA_TYPE node_total_proba = 0;

		mpi_iteration() {}
		mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

		size_t get_total_num_object(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object;
			MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			return total_num_object;
		}
		template<class T>
		T average_value(std::function<T(char const *object_begin, char const *object_end)> const &observable) const {
			return iqs::iteration::average_value(observable) / node_total_proba;
		}
		template<class T>
		T average_value(std::function<T(char const *object_begin, char const *object_end)> const &observable, MPI_Comm communicator) const {
			int size, rank;
			MPI_Comm_size(communicator, &size);
			MPI_Comm_rank(communicator, &rank);

			/* compute local average */
			T local_avg = iqs::iteration::average_value(observable);

			/* accumulate average value */
			T avg;
			MPI_Datatype avg_datatype = utils::get_mpi_datatype(avg);
			MPI_Allreduce(&local_avg, &avg, 1, avg_datatype, MPI_SUM, communicator);

			return avg;
		}
		void send_objects(size_t num_object_sent, int node, MPI_Comm communicator, bool send_num_child=false) {
			const static size_t max_int = 1 << 31 - 1;

			/* send size */
			MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

			if (num_object_sent != 0) {
				size_t begin = num_object - num_object_sent;

				/* prepare send */
				size_t send_object_begin = object_begin[begin];
				#pragma omp parallel for 
				for (size_t i = begin + 1; i <= num_object; ++i)
					object_begin[i] -= send_object_begin;

				/* send properties */
				MPI_Send(magnitude.begin() + begin, num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator);
				MPI_Send(object_begin.begin() + begin + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

				/* send objects */
				size_t send_object_size = object_begin[num_object];
				while (send_object_size > max_int) {
					MPI_Send(objects.begin() + send_object_begin, max_int, MPI_CHAR, node, 0 /* tag */, communicator);

					send_object_size -= max_int;
					send_object_begin += max_int;
				}

				MPI_Send(objects.begin() + send_object_begin, send_object_size, MPI_CHAR, node, 0 /* tag */, communicator);

				if (send_num_child) {
					/* prepare send */
					size_t child_begin = num_childs[begin];
					#pragma omp parallel for 
					for (size_t i = begin + 1; i <= num_object; ++i)
						num_childs[i] -= child_begin;

					/* send num child */
					MPI_Send(num_childs.begin() + begin + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);
				}

				/* pop */
				pop(num_object_sent, false);
			}
		}
		void receive_objects(int node, MPI_Comm communicator, bool receive_num_child=false) {
			const static size_t max_int = 1 << 31 - 1;

			/* receive size */
			size_t num_object_sent;
			MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

			if (num_object_sent != 0) {
				/* prepare state */
				resize(num_object + num_object_sent);

				/* receive properties */
				MPI_Recv(magnitude.begin() + num_object, num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);
				MPI_Recv(object_begin.begin() + num_object + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

				/* prepare receive objects */
				size_t send_object_begin = object_begin[num_object];
				size_t object_offset = send_object_begin;
				size_t send_object_size = object_begin[num_object + num_object_sent];
				allocate(send_object_begin + send_object_size);

				/* receive objects */
				while (send_object_size > max_int) {
					MPI_Recv(objects.begin() + send_object_begin, max_int, MPI_CHAR, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

					send_object_size -= max_int;
					send_object_begin += max_int;
				}
				
				MPI_Recv(objects.begin() + send_object_begin, send_object_size, MPI_CHAR, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

				/* correct values */
				#pragma omp parallel for 
				for (size_t i = num_object + 1; i <= num_object + num_object_sent; ++i)
					object_begin[i] += object_offset;

				if (receive_num_child) {
					/* receive num child */
					MPI_Recv(num_childs.begin() + num_object + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

					/* correct num child */
					size_t child_begin = num_childs[num_object];
					#pragma omp parallel for 
					for (size_t i = num_object + 1; i <= num_object + num_object_sent; ++i)
						num_childs[i] += child_begin;
				}

				num_object += num_object_sent;
			}
		}
		void equalize(MPI_Comm communicator);
		void distribute_objects(MPI_Comm communicator, int node_id);
		void gather_objects(MPI_Comm communicator, int node_id);
	};

	class mpi_symbolic_iteration : public iqs::symbolic_iteration {
		friend mpi_iteration;
		friend size_t inline get_max_num_object(mpi_it_t const &next_iteration, mpi_it_t const &last_iteration, mpi_sy_it_t const &symbolic_iteration, MPI_Comm communicator);
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, iqs::debug_t mid_step_function); 

	protected:
		iqs::utils::fast_vector<mag_t> partitioned_mag;
		iqs::utils::fast_vector<size_t> partitioned_hash;
		iqs::utils::fast_vector<bool> partitioned_is_unique;

		iqs::utils::fast_vector<mag_t> mag_buffer;
		iqs::utils::fast_vector<size_t> hash_buffer;
		iqs::utils::fast_vector<int> node_id_buffer;
		iqs::utils::fast_vector<bool> is_unique_buffer;

		void compute_collisions(MPI_Comm communicator, std::function<void()> mid_step_function=[](){});
		void mpi_resize(size_t size) {
			partitioned_mag.resize(size);
			partitioned_hash.resize(size);
			partitioned_is_unique.resize(size);
		}
		void buffer_resize(size_t size) {
			mag_buffer.resize(size);
			hash_buffer.resize(size);
			node_id_buffer.resize(size);
			is_unique_buffer.resize(size);

			if (size > next_oid_partitioner_buffer.size())
				next_oid_partitioner_buffer.resize(size);
		}

	public:
		size_t get_total_num_object(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object;
			MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			return total_num_object;
		}
		size_t get_total_num_object_after_interferences(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object_after_interference;
			MPI_Allreduce(&num_object_after_interferences, &total_num_object_after_interference, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			return total_num_object_after_interference;
		}
		mpi_symbolic_iteration() {}
	};

	/*
	for memory managment
	*/
	size_t inline get_max_num_object(mpi_it_t const &next_iteration, mpi_it_t const &last_iteration, mpi_sy_it_t const &symbolic_iteration, MPI_Comm localComm) {
		static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 2*sizeof(size_t);
		static const size_t symbolic_iteration_memory_size = (1 + 1) + (2 + 4)*sizeof(PROBA_TYPE) + (7 + 2)*sizeof(size_t) + sizeof(uint32_t) + sizeof(double) + sizeof(int);

		// get each size
		size_t next_iteration_object_size = next_iteration.objects.size();
		size_t last_iteration_object_size = last_iteration.objects.size();
		MPI_Allreduce(MPI_IN_PLACE, &next_iteration_object_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);
		MPI_Allreduce(MPI_IN_PLACE, &last_iteration_object_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);

		size_t next_iteration_property_size = next_iteration.magnitude.size();
		size_t last_iteration_property_size = last_iteration.magnitude.size();
		MPI_Allreduce(MPI_IN_PLACE, &next_iteration_property_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);
		MPI_Allreduce(MPI_IN_PLACE, &last_iteration_property_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);

		size_t symbolic_iteration_size = symbolic_iteration.magnitude.size();
		MPI_Allreduce(MPI_IN_PLACE, &symbolic_iteration_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);

		size_t last_iteration_num_object = last_iteration.num_object;
		size_t symbolic_iteration_num_object = symbolic_iteration.num_object;
		MPI_Allreduce(MPI_IN_PLACE, &last_iteration_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);
		MPI_Allreduce(MPI_IN_PLACE, &symbolic_iteration_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);

		size_t num_object_after_interferences = symbolic_iteration.num_object_after_interferences;
		MPI_Allreduce(MPI_IN_PLACE, &num_object_after_interferences, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);
		if (num_object_after_interferences == 0)
			return -1;

		// get the free memory and the total amount of memory...
		size_t free_mem;
		iqs::utils::get_free_mem(free_mem);

		// get the total memory
		size_t total_useable_memory = next_iteration_object_size + last_iteration_object_size + // size of objects
			(last_iteration_property_size + next_iteration_property_size)*iteration_memory_size + // size of properties
			symbolic_iteration_size*symbolic_iteration_memory_size + // size of symbolic properties
			free_mem; // free memory per shared memory simulation

		// compute average object size
		size_t iteration_size_per_object = 0;

		// compute the average size of an object for the next iteration:
		size_t test_size = 0;
		if (symbolic_iteration.num_object_after_interferences > 0) {
			test_size = std::max((size_t)1, (size_t)(size_average_proportion*symbolic_iteration.num_object_after_interferences));
			#pragma omp parallel for reduction(+:iteration_size_per_object)
			for (size_t oid = 0; oid < test_size; ++oid)
				iteration_size_per_object += symbolic_iteration.size[oid];
		}

		// get total average
		size_t total_test_size = std::max((size_t)1, (size_t)(size_average_proportion*symbolic_iteration.get_total_num_object_after_interferences(localComm)));
		MPI_Allreduce(MPI_IN_PLACE, &iteration_size_per_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);
		iteration_size_per_object /= total_test_size;

		// add the cost of the symbolic iteration in itself
		iteration_size_per_object += symbolic_iteration_memory_size*symbolic_iteration_num_object/last_iteration_num_object/2; // size for symbolic iteration
		
		// and the constant size per object
		iteration_size_per_object += iteration_memory_size;

		// and the cost of unused space
		iteration_size_per_object *= iqs::utils::upsize_policy;

		return total_useable_memory / iteration_size_per_object * (1 - safety_margin);
	}

	/*
	function to compute the maximum and minimum per node size
	*/
	size_t get_max_num_object_per_task(mpi_it_t const &iteration, MPI_Comm communicator) {
		size_t max_num_object_per_node;
		MPI_Allreduce(&iteration.num_object, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
		return max_num_object_per_node;
	}
	size_t get_min_num_object_per_task(mpi_it_t const &iteration, MPI_Comm communicator) {
		size_t min_num_object_per_node;
		MPI_Allreduce(&iteration.num_object, &min_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator);
		return min_num_object_per_node;
	}
	size_t get_max_num_symbolic_object_per_task(mpi_it_t const &iteration, MPI_Comm communicator) {
		size_t num_symbolic_object = iteration.get_num_symbolic_object();
		size_t max_num_symbolic_object_per_node;
		MPI_Allreduce(&num_symbolic_object, &max_num_symbolic_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
		return max_num_symbolic_object_per_node;
	}
	size_t get_min_num_symbolic_object_per_task(mpi_it_t const &iteration, MPI_Comm communicator) {
		size_t num_symbolic_object = iteration.get_num_symbolic_object();
		size_t min_num_symbolic_object_per_node;
		MPI_Allreduce(&num_symbolic_object, &min_num_symbolic_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MIN, communicator);
		return min_num_symbolic_object_per_node;
	}

	/*
	simulation function
	*/
	void simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object=0, iqs::debug_t mid_step_function=[](int){}) {
		/* get local size */
		MPI_Comm localComm;
		int rank, size, local_size;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);
		MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_size(localComm, &local_size);

		int n = 0;
		auto actual_mid_step_function = [&]() { mid_step_function(n++); };

		/* start actual simulation */
		iteration.compute_num_child(rule, actual_mid_step_function);
		actual_mid_step_function();

		/* equalize symbolic objects */
		size_t max_n_object;
		int max_equalize = iqs::utils::log_2_upper_bound(size);
		while((max_n_object = get_max_num_symbolic_object_per_task(iteration, communicator)) > min_equalize_size &&
			((float)(max_n_object - get_min_num_symbolic_object_per_task(iteration, communicator)))/((float)max_n_object) > equalize_imablance &&
			--max_equalize >= 0)
				iteration.equalize_symbolic(communicator);

		/* rest of the simulation */
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, actual_mid_step_function);
		symbolic_iteration.compute_collisions(communicator, actual_mid_step_function);

		if (max_num_object == 0)
			max_num_object = get_max_num_object(iteration_buffer, iteration, symbolic_iteration, localComm)/2;

		/* finalize simulation */
		symbolic_iteration.finalize(rule, iteration, iteration_buffer, max_num_object / local_size, actual_mid_step_function);
		actual_mid_step_function();
		std::swap(iteration_buffer, iteration);

		/* equalize and/or normalize */
		max_equalize = iqs::utils::log_2_upper_bound(size);
		while((max_n_object = get_max_num_object_per_task(iteration, communicator)) > min_equalize_size &&
			((float)(max_n_object - get_min_num_object_per_task(iteration, communicator)))/((float)max_n_object)/max_n_object > equalize_imablance &&
			--max_equalize >= 0)
				iteration.equalize(communicator); 

		/* finish by normalizing */
		iteration.normalize(communicator, actual_mid_step_function);

		MPI_Comm_free(&localComm);
	}

	/*
	distributed interference function
	*/
	void mpi_symbolic_iteration::compute_collisions(MPI_Comm communicator, std::function<void()> mid_step_function) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		if (size == 1)
			return iqs::symbolic_iteration::compute_collisions();

		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		elimination_maps.resize(num_threads);

		/*
		function to add a key
		*/
		const auto insert_key = [&](size_t const oid, robin_hood::unordered_map<size_t, size_t> &elimination_map, std::vector<int> &global_num_object_after_interferences) {
			int node_id = node_id_buffer[oid];

			/* accessing key */
			auto [it, unique] = elimination_map.insert({hash_buffer[oid], oid});
			if (unique) {
				/* increment values */
				++global_num_object_after_interferences[node_id];
				is_unique_buffer[oid] = true;
			} else {
				auto other_oid = it->second;
				auto other_node_id = node_id_buffer[other_oid];

				bool is_greater = global_num_object_after_interferences[node_id] >= global_num_object_after_interferences[other_node_id];
				if (is_greater) {
					/* if it exist add the probabilities */
					mag_buffer[other_oid] += mag_buffer[oid];
					is_unique_buffer[oid] = false;
				} else {
					/* keep this graph */
					it->second = oid;

					/* if the size aren't balanced, add the probabilities */
					mag_buffer[oid] += mag_buffer[other_oid];
					is_unique_buffer[oid] = true;
					is_unique_buffer[other_oid] = false;

					/* increment values */
					++global_num_object_after_interferences[node_id];
					--global_num_object_after_interferences[other_node_id];
				}
			}
		};

		const auto compute_interferences = [&](size_t *end_iterator, bool first) {
			size_t oid_end = std::distance(next_oid.begin(), end_iterator);
			mpi_resize(oid_end);

			int n_segment = size*num_threads;
			std::vector<int> local_disp(num_threads*(n_segment + 1), 0);
			std::vector<int> local_count(num_threads*n_segment, 0);
			std::vector<int> global_disp(num_threads*(n_segment + 1), 0);
			std::vector<int> global_count(num_threads*n_segment, 0);

			int const num_bucket = iqs::utils::nearest_power_of_two(load_balancing_bucket_per_thread*n_segment);
			std::vector<int> load_balancing_begin(n_segment + 1, 0);
			std::vector<int> partition_begin(num_threads*(num_bucket + 1), 0);
			std::vector<size_t> total_partition_begin(num_bucket + 1, 0);

			std::vector<size_t> local_load_begin(num_threads + 1, 0);
			std::vector<size_t> global_load_begin(num_threads + 1, 0);

			size_t const bitmask = num_bucket - 1;
			const auto partitioner = [&](size_t const oid) {
				return hash[oid] & bitmask;
			};
			
			local_load_begin[0] = 0; global_load_begin[0] = 0;

			#pragma omp parallel
			{
				std::vector<int> send_disp(size + 1, 0);
				std::vector<int> send_count(size, 0);
				std::vector<int> receive_disp(size + 1, 0);
				std::vector<int> receive_count(size, 0);

				std::vector<int> global_num_object_after_interferences(size, 0);
				int thread_id = omp_get_thread_num();
				auto &elimination_map = elimination_maps[thread_id];

				local_load_begin[thread_id + 1] = (thread_id + 1) * oid_end / num_threads;

				#pragma omp barrier

				size_t this_oid_begin = local_load_begin[thread_id];
				size_t this_oid_end = local_load_begin[thread_id + 1];

				int disp_offset_begin = thread_id*(n_segment + 1);
				int disp_offset_end = (thread_id + 1)*(n_segment + 1);
				int count_offset_begin = thread_id*n_segment;

				/* partition nodes */
				if (first) {
					iqs::utils::generalized_partition_from_iota(next_oid.begin() + this_oid_begin, next_oid.begin() + this_oid_end, this_oid_begin,
						partition_begin.begin() + thread_id*(num_bucket + 1), partition_begin.begin() + (thread_id + 1)*(num_bucket + 1),
						partitioner);
				} else
					iqs::utils::/*complete_*/generalized_partition(next_oid.begin() + this_oid_begin, next_oid.begin() + this_oid_end, next_oid_partitioner_buffer.begin() + this_oid_begin,
						partition_begin.begin() + thread_id*(num_bucket + 1), partition_begin.begin() + (thread_id + 1)*(num_bucket + 1),
						partitioner);

				/* compute total partition for load balancing */
				for (int i = 1; i <= num_bucket; ++i)
					#pragma omp atomic
					total_partition_begin[i] += partition_begin[(num_bucket + 1)*thread_id + i];

				/* compute load balancing */
				#pragma omp barrier
				#pragma omp single
				{

					MPI_Reduce(rank == 0 ? MPI_IN_PLACE : &total_partition_begin[1], &total_partition_begin[1],
						num_bucket, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator);

					if (rank == 0)
						iqs::utils::load_balancing_from_prefix_sum(total_partition_begin.begin(), total_partition_begin.end(),
							load_balancing_begin.begin(), load_balancing_begin.end());

					MPI_Bcast(&load_balancing_begin[1], n_segment, MPI_INT, 0, communicator);
				}

				/* rework indexes */
				for (int i = 1; i <= n_segment; ++i)
					local_disp[(n_segment + 1)*thread_id + i] = partition_begin[(num_bucket + 1)*thread_id + load_balancing_begin[i]];

				/* compute count */
				std::adjacent_difference(local_disp.begin() + disp_offset_begin + 1, local_disp.begin() + disp_offset_end, local_count.begin() + count_offset_begin);

				/* generate partitioned hash */
				for (size_t id = this_oid_begin; id < this_oid_end; ++id) {
					size_t oid = next_oid[id];

					partitioned_mag[id] = magnitude[oid];
					partitioned_hash[id] = hash[oid];
				}

				/* send partition size */
				#pragma omp for ordered
				for (int thread = 0; thread < num_threads; ++thread)
					#pragma omp ordered
					MPI_Alltoall(&local_count[count_offset_begin], num_threads, MPI_INT, &global_count[count_offset_begin], num_threads, MPI_INT, communicator);
				std::partial_sum(global_count.begin() + count_offset_begin, global_count.begin() + count_offset_begin + n_segment, global_disp.begin() + disp_offset_begin + 1);

				#pragma omp barrier

				/* rework counts */
				send_disp[0] = 0; receive_disp[0] = 0;
				for (int i = 0; i < size; ++i) {
					/* send disp and count */
					send_disp[i + 1] = local_disp[disp_offset_begin + (i + 1)*num_threads];
					send_count[i] = send_disp[i + 1] - send_disp[i];

					/* receive disp and count */
					receive_disp[i + 1] = global_disp[disp_offset_begin + (i + 1)*num_threads];
					receive_count[i] = receive_disp[i + 1] - receive_disp[i];
				}

				global_load_begin[thread_id + 1] = receive_disp[size];

				#pragma omp barrier
				#pragma omp single
				{
					__gnu_parallel::partial_sum(global_load_begin.begin() + 1, global_load_begin.begin() + num_threads + 1, global_load_begin.begin() + 1);

					/* resize */
					buffer_resize(global_load_begin[num_threads]);
				}

				size_t this_oid_buffer_begin = global_load_begin[thread_id];
				size_t this_oid_buffer_end = global_load_begin[thread_id + 1];

				/* share actual partition */
				#pragma omp for ordered
				for (int thread = 0; thread < num_threads; ++thread)
					#pragma omp ordered
					{
						MPI_Alltoallv(partitioned_hash.begin() + this_oid_begin, &send_count[0], &send_disp[0], MPI_UNSIGNED_LONG_LONG,
							hash_buffer.begin() + this_oid_buffer_begin, &receive_count[0], &receive_disp[0], MPI_UNSIGNED_LONG_LONG, communicator);
						MPI_Alltoallv(partitioned_mag.begin()  + this_oid_begin, &send_count[0], &send_disp[0], mag_MPI_Datatype,
							mag_buffer.begin() + this_oid_buffer_begin, &receive_count[0], &receive_disp[0], mag_MPI_Datatype, communicator);
					}

				/* prepare node_id buffer */
				for (int node = 0; node < size; ++node)
					for (size_t i = receive_disp[node] + this_oid_buffer_begin; i < receive_disp[node + 1] + this_oid_buffer_begin; ++i)
						node_id_buffer[i] = node;

				size_t total_size = 0;
				for (int other_thread_id = 0; other_thread_id < num_threads; ++other_thread_id)
					for (int node_id = 0; node_id < size; ++node_id)
						total_size += global_count[other_thread_id*n_segment + node_id*num_threads + thread_id];

				elimination_map.reserve(total_size);

				#pragma omp barrier

				for (int other_thread_id = 0; other_thread_id < num_threads; ++other_thread_id) {
					size_t other_oid_begin = global_load_begin[other_thread_id];

					for (int node_id = 0; node_id < size; ++node_id) {
						size_t begin = global_disp[other_thread_id*(n_segment + 1) + node_id*num_threads + thread_id] + other_oid_begin;
						size_t end = global_disp[other_thread_id*(n_segment + 1) + node_id*num_threads + thread_id + 1] + other_oid_begin;

						for (size_t i = begin; i < end; ++i)
							insert_key(i, elimination_map, global_num_object_after_interferences);
					}
				}

				elimination_map.clear();

				/* share back partition */
				#pragma omp barrier
				#pragma omp for ordered
				for (int thread = 0; thread < num_threads; ++thread)
					#pragma omp ordered
					{
						MPI_Alltoallv(mag_buffer.begin() + this_oid_buffer_begin, &receive_count[0], &receive_disp[0], mag_MPI_Datatype,
							partitioned_mag.begin() + this_oid_begin, &send_count[0], &send_disp[0], mag_MPI_Datatype, communicator);
						MPI_Alltoallv(is_unique_buffer.begin() + this_oid_buffer_begin, &receive_count[0], &receive_disp[0], MPI_CHAR,
							partitioned_is_unique.begin() + this_oid_begin, &send_count[0], &send_disp[0], MPI_CHAR, communicator);
					}
				#pragma omp barrier

				/* un-partition magnitude */
				for (size_t id = this_oid_begin; id < this_oid_end; ++id) {
					size_t oid = next_oid[id];

					is_unique[oid] = partitioned_is_unique[id];
					magnitude[oid] = partitioned_mag[id];
				}
			}

			/* keep only unique objects */
			return __gnu_parallel::partition(next_oid.begin(), end_iterator,
				[&](size_t const &oid) {
					if (!is_unique[oid])
						return false;

					return std::norm(magnitude[oid]) > tolerance;
				});
		};

		mid_step_function();

		/* !!!!!!!!!!!!!!!!
		step (4)
		 !!!!!!!!!!!!!!!! */

		bool fast = false;
		bool skip_test = collision_test_proportion == 0 || collision_tolerance == 0 || get_total_num_object(communicator) < min_collision_size;
		size_t test_size = num_object*collision_test_proportion;
		size_t *partitioned_it = next_oid.begin() + (skip_test ? num_object : test_size);

		/* get all unique graphs with a non zero probability */
		if (!skip_test) {
			size_t total_test_size;
			MPI_Allreduce(&test_size, &total_test_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			auto test_partitioned_it = compute_interferences(partitioned_it, true);

			size_t number_of_collision = std::distance(test_partitioned_it, partitioned_it);
			MPI_Allreduce(MPI_IN_PLACE, &number_of_collision, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			fast = number_of_collision < total_test_size*collision_tolerance;

			partitioned_it = std::rotate(test_partitioned_it, partitioned_it, next_oid.begin() + num_object);
		}
		if (!fast)
			partitioned_it = compute_interferences(partitioned_it, skip_test);
			
		num_object_after_interferences = std::distance(next_oid.begin(), partitioned_it);
	}

	/*
	distributed normalization function
	*/
	void mpi_iteration::normalize(MPI_Comm communicator, std::function<void()> mid_step_function) {
		mid_step_function();

		/* !!!!!!!!!!!!!!!!
		step (8)
		 !!!!!!!!!!!!!!!! */

		node_total_proba = 0;
		total_proba = 0;

		#pragma omp parallel for reduction(+:node_total_proba)
		for (size_t oid = 0; oid < num_object; ++oid)
			node_total_proba += std::norm(magnitude[oid]);

		/* accumulate probabilities on the master node */
		MPI_Allreduce(&node_total_proba, &total_proba, 1, Proba_MPI_Datatype, MPI_SUM, communicator);
		PROBA_TYPE normalization_factor = std::sqrt(total_proba);

		if (normalization_factor != 1)
			#pragma omp parallel for 
			for (size_t oid = 0; oid < num_object; ++oid)
				magnitude[oid] /= normalization_factor;

		node_total_proba /= total_proba;

		mid_step_function();
	}

	/*
	equalize symbolic object across nodes
	*/
	void mpi_iteration::equalize_symbolic(MPI_Comm communicator) {
		MPI_Request request = MPI_REQUEST_NULL;

		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		/* gather sizes */
		size_t num_symbolic_object = get_num_symbolic_object();

		size_t *sizes;
		if (rank == 0)
			sizes = (size_t*)calloc(size, sizeof(size_t));
		MPI_Gather(&num_symbolic_object, 1, MPI_UNSIGNED_LONG_LONG, sizes, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);

		/* compute pair_id*/
		int this_pair_id;
		int *pair_id = rank == 0 ? new int[size] : NULL;
		if (rank == 0)
			utils::make_equal_pairs(sizes, sizes + size, pair_id);

		/* scatter pair_id */
		MPI_Scatter(pair_id, 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
		if (rank == 0)
			delete[] pair_id;

		/* skip if this node is alone */
		if (this_pair_id == rank)
			return;

		/* get the number of objects of the respective pairs */
		size_t other_num_object;
		size_t other_max_symbolic_object_size;
		
		MPI_Isend(&num_symbolic_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, &request);
		MPI_Isend(&max_symbolic_object_size, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, &request);

		MPI_Recv(&other_num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, MPI_STATUS_IGNORE);
		MPI_Recv(&other_max_symbolic_object_size, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

		max_symbolic_object_size = std::max(max_symbolic_object_size, other_max_symbolic_object_size);

		/* equalize amoung pairs */
		if (num_symbolic_object > other_num_object) {
			size_t num_symbolic_object_to_send = (num_symbolic_object - other_num_object) / 2;
			size_t num_object_sent = num_object - 
				std::distance(num_childs.begin(),
					std::upper_bound(num_childs.begin(), num_childs.begin() + num_object,
						num_symbolic_object - num_symbolic_object_to_send));

			send_objects(num_object_sent, this_pair_id, communicator, true);
		} else if (num_symbolic_object < other_num_object)
			receive_objects(this_pair_id, communicator, true);
	}

	/*
	"utility" functions from here on:
	*/
	/*
	equalize the number of objects across nodes
	*/
	void mpi_iteration::equalize(MPI_Comm communicator) {
		MPI_Request request = MPI_REQUEST_NULL;

		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		/* gather sizes */
		size_t *sizes;
		if (rank == 0)
			sizes = (size_t*)calloc(size, sizeof(size_t));
		MPI_Gather(&num_object, 1, MPI_UNSIGNED_LONG_LONG, sizes, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);

		/* compute pair_id*/
		int this_pair_id;
		int *pair_id = rank == 0 ? new int[size] : NULL;
		if (rank == 0)
			utils::make_equal_pairs(sizes, sizes + size, pair_id);

		/* scatter pair_id */
		MPI_Scatter(pair_id, 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
		if (rank == 0)
			delete[] pair_id;

		/* skip if this node is alone */
		if (this_pair_id == rank)
			return;

		/* get the number of objects of the respective pairs */
		size_t other_num_object;
		MPI_Isend(&num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, &request);
		MPI_Isend(&num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, &request);
		
		MPI_Recv(&other_num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, MPI_STATUS_IGNORE);
		MPI_Recv(&other_num_object, 1, MPI_UNSIGNED_LONG_LONG, this_pair_id, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

		/* equalize amoung pairs */
		if (num_object > other_num_object) {
			size_t num_object_sent = (num_object -  other_num_object) / 2;
			send_objects(num_object_sent, this_pair_id, communicator);
		} else if (num_object < other_num_object)
			receive_objects(this_pair_id, communicator);
	}

	/*
	function to distribute objects across nodes
	*/
	void mpi_iteration::distribute_objects(MPI_Comm communicator, int node_id=0) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		size_t initial_num_object = num_object;
		if (rank == node_id) {
			for (int node = 1; node < size; ++node) {
				int node_to_send = node <= node_id ? node - 1 : node; //skip this node
				size_t num_object_sent = (initial_num_object * (node + 1)) / size - (initial_num_object * node) / size; //better way to spread evently

				/* send objects */
				send_objects(num_object_sent, node_to_send, communicator);
			}

		} else
			/* receive objects */
			receive_objects(node_id, communicator);
	}

	/*
	function to gather object from all nodes
	*/
	void mpi_iteration::gather_objects(MPI_Comm communicator, int node_id=0) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		if (rank == node_id) {
			for (int node = 1; node < size; ++node) {
				int receive_node = node <= node_id ? node - 1 : node;

				/* receive objects */
				receive_objects(receive_node, communicator);
			}

		} else
			/* send objects */
			send_objects(num_object, node_id, communicator);

		/* compute node_total_proba */
		node_total_proba = rank == node_id;
	}
}