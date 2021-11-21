#pragma once

#include "iqs.hpp"

#include <mpi.h>

#ifndef MIN_EQUALIZE_SIZE
	#define MIN_EQUALIZE_SIZE 1000
#endif
#ifndef EQUALIZE_IMBALANCE
	#define EQUALIZE_IMBALANCE 0.2
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
		void normalize(MPI_Comm communicator);

	public:
		PROBA_TYPE node_total_proba = 0;

		mpi_iteration() {}
		mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

		size_t get_total_num_object(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object;
			MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG, MPI_SUM, communicator);

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
		void send_objects(size_t num_object_sent, int node, MPI_Comm communicator) {
			/* send size */
			MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG, node, 0 /* tag */, communicator);

			if (num_object_sent != 0) {
				size_t begin = num_object - num_object_sent;

				/* prepare send */
				size_t send_object_begin = object_begin[begin];
				#pragma omp parallel for schedule(static)
				for (size_t i = begin + 1; i <= num_object; ++i)
					object_begin[i] -= send_object_begin;

				/* send properties */
				MPI_Send(magnitude.begin() + begin, num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator);
				MPI_Send(object_begin.begin() + begin + 1, num_object_sent, MPI_UNSIGNED_LONG, node, 0 /* tag */, communicator);

				/* send objects */
				const size_t max_int = 0x7FFFFFFF;
				size_t send_object_size = object_begin[num_object];
				while (send_object_size > max_int) {
					MPI_Send(objects.begin() + send_object_begin, max_int, MPI_CHAR, node, 0 /* tag */, communicator);
					send_object_size -= max_int;
					send_object_begin += max_int;
				}
				MPI_Send(objects.begin() + send_object_begin, send_object_size, MPI_CHAR, node, 0 /* tag */, communicator);

				/* pop */
				pop(num_object_sent, false);
			}
		}
		void receive_objects(int node, MPI_Comm communicator) {
			/* receive size */
			size_t num_object_sent;
			MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG, node, 0 /* tag */, communicator, &utils::global_status);

			if (num_object_sent != 0) {
				/* prepare state */
				resize(num_object + num_object_sent);

				/* receive properties */
				MPI_Recv(magnitude.begin() + num_object, num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator, &utils::global_status);
				MPI_Recv(object_begin.begin() + num_object + 1, num_object_sent, MPI_UNSIGNED_LONG, node, 0 /* tag */, communicator, &utils::global_status);

				/* prepare receive objects */
				size_t send_object_begin = object_begin[num_object];
				size_t send_object_size = object_begin[num_object + num_object_sent];
				allocate(send_object_begin + send_object_size);

				/* receive objects */
				const size_t max_int = 0x7FFFFFFF;
				while (send_object_size > max_int) {
					MPI_Recv(objects.begin() + send_object_begin, max_int, MPI_CHAR, node, 0 /* tag */, communicator, &utils::global_status);
					send_object_size -= max_int;
					send_object_begin += max_int;
				}
				MPI_Recv(objects.begin() + send_object_begin, send_object_size, MPI_CHAR, node, 0 /* tag */, communicator, &utils::global_status);

				/* correct values */
				#pragma omp parallel for schedule(static)
				for (size_t i = num_object + 1; i <= num_object + num_object_sent; ++i)
					object_begin[i] += send_object_begin;
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
		iqs::utils::fast_vector/*numa_vector*/<mag_t> partitioned_mag;
		iqs::utils::fast_vector/*numa_vector*/<size_t> partitioned_hash;
		iqs::utils::fast_vector/*numa_vector*/<bool> partitioned_is_unique;

		iqs::utils::fast_vector/*numa_vector*/<mag_t> mag_buffer;
		iqs::utils::fast_vector/*numa_vector*/<size_t> hash_buffer;
		iqs::utils::fast_vector/*numa_vector*/<bool> is_unique_buffer;
		iqs::utils::fast_vector/*numa_vector*/<int> node_id_buffer;

		void compute_collisions(MPI_Comm communicator);
		void mpi_resize(size_t size) {
			partitioned_mag.zero_resize(size);
			partitioned_hash.zero_resize(size);
			partitioned_is_unique.zero_resize(size);
		}
		void buffer_resize(size_t size) {
			mag_buffer.zero_resize(size);
			hash_buffer.zero_resize(size);
			is_unique_buffer.zero_resize(size);
			node_id_buffer.zero_resize(size);
		}

		long long int memory_size = (1 + 3) + (2 + 4)*sizeof(PROBA_TYPE) + (6 + 4)*sizeof(size_t) + sizeof(uint32_t) + sizeof(double) + sizeof(int);

	public:
		size_t get_total_num_object(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object;
			MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG, MPI_SUM, communicator);

			return total_num_object;
		}
		size_t get_total_num_object_after_interferences(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object_after_interference;
			MPI_Allreduce(&num_object_after_interferences, &total_num_object_after_interference, 1, MPI_UNSIGNED_LONG, MPI_SUM, communicator);

			return total_num_object_after_interference;
		}
		mpi_symbolic_iteration() {}
	};

	/*
	for memory managment
	*/
	size_t inline get_max_num_object(mpi_it_t const &next_iteration, mpi_it_t const &last_iteration, mpi_sy_it_t const &symbolic_iteration, MPI_Comm communicator) {
		// get the shared memory communicator
		MPI_Comm localComm;
		int rank, local_rank, local_size;
		MPI_Comm_rank(communicator, &rank);
		MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_size(localComm, &local_size);
		MPI_Comm_rank(localComm, &local_rank);

		// get the free memory and the total amount of memory...
		size_t free_mem;
		iqs::utils::get_free_mem(free_mem);

		// get each size
		size_t next_iteration_object_size = next_iteration.objects.size();
		size_t last_iteration_object_size = last_iteration.objects.size();
		MPI_Allreduce(MPI_IN_PLACE, &next_iteration_object_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);
		MPI_Allreduce(MPI_IN_PLACE, &last_iteration_object_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);

		size_t next_iteration_property_size = next_iteration.magnitude.size();
		size_t last_iteration_property_size = last_iteration.magnitude.size();
		MPI_Allreduce(MPI_IN_PLACE, &next_iteration_property_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);
		MPI_Allreduce(MPI_IN_PLACE, &last_iteration_property_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);

		size_t symbolic_iteration_size = symbolic_iteration.magnitude.size();
		MPI_Allreduce(MPI_IN_PLACE, &symbolic_iteration_size, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);

		size_t last_iteration_num_object = last_iteration.num_object;
		size_t symbolic_iteration_num_object = symbolic_iteration.num_object;
		MPI_Allreduce(MPI_IN_PLACE, &last_iteration_num_object, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);
		MPI_Allreduce(MPI_IN_PLACE, &symbolic_iteration_num_object, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);

		// get the total memory
		size_t total_useable_memory = next_iteration_object_size + last_iteration_object_size + // size of objects
			last_iteration_property_size*last_iteration.memory_size + next_iteration_property_size*next_iteration.memory_size + // size of properties
			symbolic_iteration_size*symbolic_iteration.memory_size + // size of symbolic properties
			free_mem; // free memory per shared memory simulation

		// compute average object size
		size_t iteration_size_per_object = 0;

		// compute the average size of an object for the next iteration:
		size_t test_size = std::max((size_t)1, (size_t)(size_average_proportion*symbolic_iteration.num_object_after_interferences));
		#pragma omp parallel for reduction(+:iteration_size_per_object)
		for (size_t oid = 0; oid < test_size; ++oid)
			iteration_size_per_object += symbolic_iteration.size[oid];

		// get total average
		size_t total_test_size = std::max((size_t)1, (size_t)(size_average_proportion*symbolic_iteration.get_total_num_object_after_interferences(localComm)));
		MPI_Allreduce(MPI_IN_PLACE, &iteration_size_per_object, 1, MPI_UNSIGNED_LONG, MPI_SUM, localComm);
		iteration_size_per_object /= total_test_size;

		// add the cost of the symbolic iteration in itself
		iteration_size_per_object += symbolic_iteration.memory_size*symbolic_iteration_num_object/last_iteration_num_object/2; // size for symbolic iteration
		
		// and the constant size per object
		iteration_size_per_object += next_iteration.memory_size;

		// and the cost of unused space
		iteration_size_per_object *= iqs::utils::upsize_policy;

		return total_useable_memory / iteration_size_per_object * (1 - safety_margin) / local_size;
	}

	/*
	function to compute the maximum per node size imbablance
	*/
	float get_max_num_object_imbalance(mpi_it_t const &iteration, size_t const size_comp, MPI_Comm communicator) {
		size_t total_imbalance, local_imbalance = (size_t)std::abs((long long int)iteration.num_object - (long long int)size_comp);
		MPI_Allreduce(&local_imbalance, &total_imbalance, 1, MPI_UNSIGNED_LONG, MPI_MAX, communicator);
		return ((float) total_imbalance) / ((float) size_comp);
	}

	/*
	simulation function
	*/
	void simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object=0, iqs::debug_t mid_step_function=[](int){}) {
		int size;
		MPI_Comm_size(communicator, &size);

		/* actual simulation */
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
		symbolic_iteration.compute_collisions(communicator);

		if (max_num_object == 0)
			max_num_object = get_max_num_object(iteration_buffer, iteration, symbolic_iteration, communicator)/2;

		symbolic_iteration.finalize(rule, iteration, iteration_buffer, max_num_object, mid_step_function);
		std::swap(iteration_buffer, iteration);

		/* equalize and/or normalize */
		size_t average_num_object = iteration.get_total_num_object(communicator)/size;
		size_t max_num_object_per_node;
		MPI_Allreduce(&iteration.num_object, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG, MPI_MAX, communicator);
		if (max_num_object_per_node > min_equalize_size) {

			/* if both condition are met equalize */
			float max_imbalance = get_max_num_object_imbalance(iteration, average_num_object, communicator);
			if (max_imbalance > equalize_imablance)
				do {
					iteration.equalize(communicator);
				} while(get_max_num_object_imbalance(iteration, average_num_object, communicator) > equalize_imablance);
		}
			
		mid_step_function(8);

		/* finish by normalizing */
		iteration.normalize(communicator);

		mid_step_function(9);
	}

	/*
	distributed interference function
	*/
	void mpi_symbolic_iteration::compute_collisions(MPI_Comm communicator) {
		int size, rank;

		/*
		function for partition
		*/
		auto static const partitioner = [&](size_t const &oid) {
			/* check if graph is unique */
			if (!is_unique[oid])
				return false;

			return std::norm(magnitude[oid]) > tolerance;
		};

		/*
		function to add a key
		*/
		auto static const insert_key = [&](size_t oid, robin_hood::unordered_map<size_t, size_t> &elimination_map, int *global_num_object_after_interferences) {
			int node_id = node_id_buffer[oid];

			/* accessing key */
			auto [it, unique] = elimination_map.insert({hash_buffer[oid], oid});
			if (unique) {
				/* keep this graph */
				is_unique_buffer[oid] = true;

				/* increment values */
				++global_num_object_after_interferences[node_id];
			} else {
				auto other_oid = it->second;
				auto other_node_id = node_id_buffer[other_oid];

				bool is_greater = global_num_object_after_interferences[node_id] >= global_num_object_after_interferences[other_node_id];
				if (is_greater) {
					/* if it exist add the probabilities */
					mag_buffer[other_oid] += mag_buffer[oid];

					/* discard this graph */
					is_unique_buffer[oid] = false;
				} else {
					/* keep this graph */
					it->second = oid;

					/* if the size aren't balanced, add the probabilities */
					mag_buffer[oid] += mag_buffer[other_oid];

					/* discard the other graph */
					is_unique_buffer[oid] = true;
					is_unique_buffer[other_oid] = false;

					/* increment values */
					++global_num_object_after_interferences[node_id];
					--global_num_object_after_interferences[other_node_id];
				}
			}
		};

		/* !!!!!!!!!!!!!!!!
		step (4)

		Actual code :
		 !!!!!!!!!!!!!!!! */

		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		if (size == 1)
			return iqs::symbolic_iteration::compute_collisions();

		const int num_bucket = num_threads*size > 1 ? iqs::utils::nearest_power_of_two(load_balancing_bucket_per_thread*num_threads*size) : 1;

		/* prepare buffers */
		size_t *local_disp = new size_t[num_bucket + 1]();
		int *local_count = new int[num_bucket];
		int *send_disp = new int [size + 1]();
		int *send_count = new int [size];
		int *receive_disp = new int[size + 1]();
		int *receive_count = new int[size];
		int *partition_begin = new int[num_threads*size + 1]();
		int *send_partition_count = new int[num_threads*size + 1]();
		int *load_balancing_begin = new int[size*num_threads + 1]();
		size_t *total_disp = rank == 0 ? new size_t[num_bucket + 1]() : NULL;

		mpi_resize(num_object);

		/* partition nodes */
		iqs::utils::generalized_modulo_partition_power_of_two((size_t)0, num_object,
			next_oid.begin(), hash.begin(),
			local_disp, num_bucket);
		__gnu_parallel::adjacent_difference(local_disp + 1, local_disp + num_bucket + 1, local_count, std::minus<size_t>());

		/* get node list */
		MPI_Reduce(local_disp + 1, total_disp + 1, num_bucket, MPI_UNSIGNED_LONG, MPI_SUM, 0, communicator);

		/* compute load sharing */
		if (rank == 0)
			iqs::utils::load_balancing_from_prefix_sum(total_disp, total_disp + num_bucket + 1,
				load_balancing_begin, load_balancing_begin + size*num_threads + 1);

		/* broadcast load sharing */
		MPI_Bcast(load_balancing_begin + 1, size*num_threads, MPI_INT, 0, communicator);

		/* rework counts */
		for (int i = 0; i < num_threads*size; ++i) {
			int begin = load_balancing_begin[i];
			int end = load_balancing_begin[i + 1];

			send_partition_count[i] = local_disp[end] - local_disp[begin];
		}
		for (int i = 0; i < size; ++i) {
			int end = load_balancing_begin[(i + 1)*num_threads];

			/* displacement */
			send_disp[i + 1] = local_disp[end];
			send_count[i] = send_disp[i + 1] - send_disp[i];
		}

		/* get global count and disp */
		MPI_Alltoall(send_count, 1, MPI_INT, receive_count, 1, MPI_INT, communicator);
		receive_disp[0] = 0;
		__gnu_parallel::partial_sum(receive_count, receive_count + size + 1, receive_disp + 1);

		/* resize and prepare node_id buffer */
		size_t global_num_object = receive_disp[size];
		buffer_resize(global_num_object);
		for (int node = 0; node < size; ++node)
			#pragma omp parallel for
			for (size_t i = receive_disp[node]; i < receive_disp[node + 1]; ++i)
				node_id_buffer[i] = node;

		/* generate partitioned hash */
		#pragma omp parallel for schedule(static)
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = next_oid[id];

			partitioned_mag[id] = magnitude[oid];
			partitioned_hash[id] = hash[oid];
		}
			
		/* share hash and magnitude */
		MPI_Alltoallv(partitioned_hash.begin(), send_count, send_disp, MPI_UNSIGNED_LONG,
			hash_buffer.begin(), receive_count, receive_disp, MPI_UNSIGNED_LONG, communicator);
		MPI_Alltoallv(partitioned_mag.begin(), send_count, send_disp, mag_MPI_Datatype,
			mag_buffer.begin(), receive_count, receive_disp, mag_MPI_Datatype, communicator);

		/* share partitions */
		MPI_Alltoall(send_partition_count, num_threads, MPI_INT,
			partition_begin + 1, num_threads, MPI_INT, communicator);
		__gnu_parallel::partial_sum(partition_begin + 1, partition_begin + size*num_threads + 1, partition_begin + 1);


		#pragma omp parallel
		{
			int *global_num_object_after_interferences = new int[size]();

			int thread_id = omp_get_thread_num();
			auto &elimination_map = elimination_maps[thread_id];

			int total_size = 0;
			for (int i = 0; i < size; ++i) {
				int partition = i*num_threads + thread_id;

				size_t begin = partition_begin[partition];
				size_t end = partition_begin[partition + 1];
				total_size += end - begin;
			}

			/* reserve hashmap */
			elimination_map.reserve(total_size);

			for (int i = 0; i < size; ++i) {
				int partition = i*num_threads + thread_id;

				size_t begin = partition_begin[partition];
				size_t end = partition_begin[partition + 1];

				/* insert into hashmap */
				for (size_t oid = begin; oid < end; ++oid)
					insert_key(oid, elimination_map, global_num_object_after_interferences);
			}

			/* clear hashmap */
			elimination_map.clear();
			delete[] global_num_object_after_interferences;
		}

		/* share is_unique and magnitude */
		MPI_Alltoallv(is_unique_buffer.begin(), receive_count, receive_disp, MPI_CHAR,
			partitioned_is_unique.begin(), send_count, send_disp, MPI_CHAR, communicator);
		MPI_Alltoallv(mag_buffer.begin(), receive_count, receive_disp, mag_MPI_Datatype,
			partitioned_mag.begin(), send_count, send_disp, mag_MPI_Datatype, communicator);

		/* regenerate real, imag and is_unique */
		#pragma omp parallel for schedule(static)
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = next_oid[id];

			magnitude[oid] = partitioned_mag[id];
			is_unique[oid] = partitioned_is_unique[id];
		}

		/* keep only unique objects */
		auto partitioned_it = __gnu_parallel::partition(next_oid.begin(), next_oid.begin() + num_object, partitioner);
		num_object_after_interferences = std::distance(next_oid.begin(), partitioned_it);

		/* free objects */
		delete[] local_disp;
		delete[] local_count;
		delete[] send_disp;
		delete[] send_count;
		delete[] receive_disp;
		delete[] receive_count;
		delete[] partition_begin;
		delete[] send_partition_count;
		delete[] load_balancing_begin;
		if (rank == 0)
			delete[] total_disp;
	}

	/*
	distributed normalization function
	*/
	void mpi_iteration::normalize(MPI_Comm communicator) {
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
	}

	/*
	"utility" functions from here on:
	*/
	/*
	equalize the number of objects across nodes
	*/
	void mpi_iteration::equalize(MPI_Comm communicator) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		/* gather sizes */
		size_t *sizes;
		if (rank == 0)
			sizes = (size_t*)calloc(size, sizeof(size_t));
		MPI_Gather(&num_object, 1, MPI_LONG_LONG_INT, sizes, 1, MPI_LONG_LONG_INT, 0, communicator);

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
		if (rank < this_pair_id) {
			MPI_Send(&num_object, 1, MPI_LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator);
			MPI_Recv(&other_num_object, 1, MPI_LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator, &utils::global_status);
		} else {
			MPI_Recv(&other_num_object, 1, MPI_LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator, &utils::global_status);
			MPI_Send(&num_object, 1, MPI_LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator);
		}

		/* equalize amoung pairs */
		if (num_object > other_num_object) {
			size_t num_object_sent = (num_object -  other_num_object) / 2;
			send_objects(num_object_sent, this_pair_id, communicator);
		} else if (num_object < other_num_object) {
			receive_objects(this_pair_id, communicator);
		}
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