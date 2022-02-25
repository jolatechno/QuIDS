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
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, iqs::debug_t mid_step_function);

	protected:
		void equalize_symbolic(MPI_Comm communicator);
		void normalize(MPI_Comm communicator, iqs::debug_t mid_step_function=[](const char*){});



		/*
		utils functions
		*/
		size_t inline get_mem_size(MPI_Comm communicator) const {
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t);

			size_t total_size, local_size = iteration_memory_size*magnitude.size() + objects.size();
			MPI_Allreduce(&local_size, &total_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_size;
		}
		float inline get_average_num_child(MPI_Comm communicator) const {
			size_t total_num_child = 0;
			#pragma omp parallel for reduction(+:total_num_child)
			for (size_t i = 0; i < truncated_num_object; ++i)
				total_num_child += num_childs[truncated_oid[i]];

			MPI_Allreduce(MPI_IN_PLACE, &total_num_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return (float)total_num_child / (float)get_total_truncated_num_object(communicator);
		}
		float inline get_average_object_size(MPI_Comm communicator) const {
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t);

			size_t total_object_size;
			MPI_Allreduce(&object_begin[num_object], &total_object_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return (float)iteration_memory_size + (float)total_object_size/(float)get_total_num_object(communicator);
		}
		size_t inline get_total_truncated_num_object(MPI_Comm communicator) const {
			size_t total_truncated_num_object;
			MPI_Allreduce(&truncated_num_object, &total_truncated_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_truncated_num_object;
		}


		/*
		function to compute the maximum and minimum per node size
		*/
		float get_avg_num_symbolic_object_per_task(MPI_Comm communicator) {
			size_t total_num_object_per_node = get_num_symbolic_object();
			MPI_Allreduce(MPI_IN_PLACE, &total_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			int size;
			MPI_Comm_size(communicator, &size);

			return (float)total_num_object_per_node/(float)size;
		}
		size_t get_max_num_symbolic_object_per_task(MPI_Comm communicator) {
			size_t max_num_object_per_node = get_num_symbolic_object();
			MPI_Allreduce(MPI_IN_PLACE, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
			return max_num_object_per_node;
		}
		size_t get_max_num_object_per_task(MPI_Comm communicator) {
			size_t max_num_object_per_node;
			MPI_Allreduce(&num_object, &max_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, communicator);
			return max_num_object_per_node;
		}


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
				MPI_Send(&magnitude[begin], num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator);
				MPI_Send(&object_begin[begin + 1], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

				/* send objects */
				size_t send_object_size = object_begin[num_object];
				while (send_object_size > max_int) {
					MPI_Send(&objects[send_object_begin], max_int, MPI_CHAR, node, 0 /* tag */, communicator);

					send_object_size -= max_int;
					send_object_begin += max_int;
				}

				MPI_Send(&objects[send_object_begin], send_object_size, MPI_CHAR, node, 0 /* tag */, communicator);

				if (send_num_child)
					/* send num child */
					MPI_Send(&num_childs[begin], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

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
				MPI_Recv(&magnitude[num_object], num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);
				MPI_Recv(&object_begin[num_object + 1], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

				/* prepare receive objects */
				size_t send_object_begin = object_begin[num_object];
				size_t object_offset = send_object_begin;
				size_t send_object_size = object_begin[num_object + num_object_sent];
				allocate(send_object_begin + send_object_size);

				/* receive objects */
				while (send_object_size > max_int) {
					MPI_Recv(&objects[send_object_begin], max_int, MPI_CHAR, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

					send_object_size -= max_int;
					send_object_begin += max_int;
				}
				
				MPI_Recv(&objects[send_object_begin], send_object_size, MPI_CHAR, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

				/* correct values */
				#pragma omp parallel for 
				for (size_t i = num_object + 1; i <= num_object + num_object_sent; ++i)
					object_begin[i] += object_offset;

				if (receive_num_child)
					/* receive num child */
					MPI_Recv(&num_childs[num_object], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

				num_object += num_object_sent;
			}
		}
		void equalize(MPI_Comm communicator);
		void distribute_objects(MPI_Comm communicator, int node_id);
		void gather_objects(MPI_Comm communicator, int node_id);
	};

	class mpi_symbolic_iteration : public iqs::symbolic_iteration {
		friend mpi_iteration;
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, iqs::debug_t mid_step_function); 

	protected:
		iqs::utils::fast_vector<mag_t> partitioned_mag;
		iqs::utils::fast_vector<size_t> partitioned_hash;
		iqs::utils::fast_vector<char /*bool*/> partitioned_is_unique;

		iqs::utils::fast_vector<mag_t> mag_buffer;
		iqs::utils::fast_vector<size_t> hash_buffer;
		iqs::utils::fast_vector<int> node_id_buffer;
		iqs::utils::fast_vector<char /*bool*/> is_unique_buffer;

		void compute_collisions(MPI_Comm communicator, iqs::debug_t mid_step_function=[](const char*){});
		void mpi_resize(size_t size) {
			#pragma omp parallel sections
			{
				#pragma omp section
				partitioned_mag.resize(size);

				#pragma omp section
				partitioned_hash.resize(size);

				#pragma omp section
				partitioned_is_unique.resize(size);
			}
		}
		void buffer_resize(size_t size) {
			#pragma omp parallel sections
			{
				#pragma omp section
				mag_buffer.resize(size);

				#pragma omp section
				hash_buffer.resize(size);

				#pragma omp section
				node_id_buffer.resize(size);

				#pragma omp section
				is_unique_buffer.resize(size);

				#pragma omp section
				if (size > next_oid_partitioner_buffer.size())
					next_oid_partitioner_buffer.resize(size);
			}
		}



		/*
		utils functions
		*/
		float inline get_average_object_size(MPI_Comm communicator) const {
			static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);

			static const size_t symbolic_iteration_memory_size = (1 + 1) + (2 + 2)*sizeof(PROBA_TYPE) + (5 + 1)*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);

			static const size_t mpi_symbolic_iteration_memory_size = 1 + 2*sizeof(PROBA_TYPE) + 1*sizeof(size_t) + sizeof(int);
			return (float)symbolic_iteration_memory_size + (float)mpi_symbolic_iteration_memory_size + hash_map_size;
		}
		size_t inline get_mem_size(MPI_Comm communicator) const {
			static const float hash_map_size = HASH_MAP_OVERHEAD*2*sizeof(size_t);

			size_t memory_size = 0;
			for (auto &map : elimination_maps)
				memory_size += elimination_maps.capacity();
			memory_size *= hash_map_size;

			static const size_t symbolic_iteration_memory_size = (1 + 1) + (2 + 2)*sizeof(PROBA_TYPE) + (5 + 1)*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);
			memory_size += magnitude.size()*symbolic_iteration_memory_size;

			static const size_t mpi_symbolic_iteration_memory_size = 1 + 2*sizeof(PROBA_TYPE) + 1*sizeof(size_t) + sizeof(int);
			memory_size += mag_buffer.size()*mpi_symbolic_iteration_memory_size;

			size_t total_size = mpi_symbolic_iteration_memory_size*magnitude.size();
			MPI_Allreduce(MPI_IN_PLACE, &total_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_size;
		}
		float get_average_child_size(MPI_Comm communicator) const {
			size_t total_size = 0;
			#pragma omp parallel for reduction(+:total_size)
			for (size_t i = 0; i < next_iteration_num_object; ++i)
				total_size += size[next_oid[i]];

			MPI_Allreduce(MPI_IN_PLACE, &total_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t);
			return (float)iteration_memory_size + (float)total_size / (float)get_total_next_iteration_num_object(communicator);
		}
		size_t inline get_total_next_iteration_num_object(MPI_Comm communicator) const {
			size_t total_next_iteration_num_object;
			MPI_Allreduce(&next_iteration_num_object, &total_next_iteration_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_next_iteration_num_object;
		}

	public:
		size_t inline get_total_num_object(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object;
			MPI_Allreduce(&num_object, &total_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			return total_num_object;
		}
		size_t inline get_total_num_object_after_interferences(MPI_Comm communicator) const {
			/* accumulate number of node */
			size_t total_num_object_after_interference;
			MPI_Allreduce(&num_object_after_interferences, &total_num_object_after_interference, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			return total_num_object_after_interference;
		}
		mpi_symbolic_iteration() {}
	};

	/*
	simulation function
	*/
	void simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object=0, iqs::debug_t mid_step_function=[](const char*){}) {
		/* get local size */
		MPI_Comm localComm;
		int rank, size, local_size;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);
		MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_size(localComm, &local_size);


		auto const get_max_num_object_initial = [&]() {
			MPI_Barrier(localComm);
			return (float)(iqs::utils::get_free_mem() + next_iteration.get_mem_size(localComm) + symbolic_iteration.get_mem_size(localComm)) /
				(iteration.get_average_object_size(localComm) + iteration.get_average_num_child(localComm)*symbolic_iteration.get_average_object_size(localComm)) *
				(1 - safety_margin)/iqs::utils::upsize_policy;
		};
		auto const get_max_num_object_final = [&]() {
			MPI_Barrier(localComm);
			return (float)(iqs::utils::get_free_mem() + next_iteration.get_mem_size(localComm)) /
				symbolic_iteration.get_average_child_size(localComm) *
				(1 - safety_margin)/iqs::utils::upsize_policy;
		};





		

		if (size == 1)
			return iqs::simulate(iteration, rule, next_iteration, symbolic_iteration, max_num_object, mid_step_function);

		/* start actual simulation */
		iteration.compute_num_child(rule, mid_step_function);
		iteration.truncated_num_object = iteration.num_object;

		/* equalize symbolic objects */
		mid_step_function("equalize_child");
		size_t max_n_object;
		int max_equalize = iqs::utils::log_2_upper_bound(size);
		while((max_n_object = iteration.get_max_num_symbolic_object_per_task(communicator)) > min_equalize_size &&
			((float)max_n_object - iteration.get_avg_num_symbolic_object_per_task(communicator))/(float)max_n_object > equalize_imablance &&
			--max_equalize >= 0) {
				iteration.equalize_symbolic(communicator);
				iteration.truncated_num_object = iteration.num_object;
			}

		/* max_num_object */
		mid_step_function("truncate");
		if (max_num_object == 0) {
			size_t max_truncated_num_object;
			do {
				max_truncated_num_object = get_max_num_object_initial();
				iteration.truncate(max_truncated_num_object/local_size);
			} while (iteration.get_total_truncated_num_object(localComm) > max_truncated_num_object*iqs::utils::upsize_policy);
		} else
			iteration.truncate(max_num_object/local_size);

		/* downsize if needed */
		if (iteration.num_object > 0) {
			if (iteration.truncated_num_object < next_iteration.num_object)
				next_iteration.resize(iteration.truncated_num_object);
			size_t next_object_size = iteration.truncated_num_object*iteration.get_object_length()/iteration.num_object;
			if (next_object_size < next_iteration.objects.size())
				next_iteration.allocate(next_object_size);
		}

		/* rest of the simulation */
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
		symbolic_iteration.compute_collisions(communicator, mid_step_function);
		symbolic_iteration.next_iteration_num_object = symbolic_iteration.num_object_after_interferences;

		/* second max_num_object */
		mid_step_function("truncate");
		if (max_num_object == 0) {
			size_t max_truncated_num_object;
			do {
				max_truncated_num_object = get_max_num_object_final();
				symbolic_iteration.truncate(max_truncated_num_object/local_size);
			} while (symbolic_iteration.get_total_next_iteration_num_object(localComm) > max_truncated_num_object*iqs::utils::upsize_policy);
		} else
			symbolic_iteration.truncate(max_num_object/local_size);

		/* finalize simulation */
		symbolic_iteration.finalize(rule, iteration, next_iteration, mid_step_function);
		mid_step_function("equalize");

		/* equalize and/or normalize */
		max_equalize = iqs::utils::log_2_upper_bound(size);
		while((max_n_object = next_iteration.get_max_num_object_per_task(communicator)) > min_equalize_size &&
			((float)max_n_object - (float)next_iteration.get_total_num_object(communicator)/(float)size)/(float)max_n_object > equalize_imablance &&
			--max_equalize >= 0)
				next_iteration.equalize(communicator); 

		/* finish by normalizing */
		next_iteration.normalize(communicator, mid_step_function);

		MPI_Comm_free(&localComm);
	}

	/*
	distributed interference function
	*/
	void mpi_symbolic_iteration::compute_collisions(MPI_Comm communicator, iqs::debug_t mid_step_function) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		if (size == 1)
			return iqs::symbolic_iteration::compute_collisions(mid_step_function);

		int num_threads;
		#pragma omp parallel
		#pragma omp single
		num_threads = omp_get_num_threads();

		elimination_maps.resize(num_threads);

		int const n_segment = size*num_threads;
		int const num_bucket = iqs::utils::nearest_power_of_two(load_balancing_bucket_per_thread*n_segment);
		size_t const offset = 8*sizeof(size_t) - iqs::utils::log_2_upper_bound(num_bucket);

		std::vector<int> load_balancing_begin(n_segment + 1, 0);
		std::vector<size_t> partition_begin(num_bucket + 1, 0);

		std::vector<int> local_disp(n_segment + 1);
		std::vector<int> local_count(n_segment);
		std::vector<int> global_disp(n_segment + 1, 0);
		std::vector<int> global_count(n_segment);

		std::vector<int> send_disp(size + 1);
		std::vector<int> send_count(size);
		std::vector<int> receive_disp(size + 1);
		std::vector<int> receive_count(size);

		mid_step_function("compute_collisions - prepare");
		mpi_resize(num_object);



			

		/* !!!!!!!!!!!!!!!!
		partition
		!!!!!!!!!!!!!!!! */
		iqs::utils::parallel_generalized_partition_from_iota(&next_oid[0], &next_oid[0] + num_object, 0,
			partition_begin.begin(), partition_begin.end(),
			[&](size_t const oid) {
				return hash[oid] >> offset;
			});

		/* generate partitioned hash */
		#pragma omp parallel for
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = next_oid[id];

			partitioned_mag[id] = magnitude[oid];
			partitioned_hash[id] = hash[oid];
		}







		/* !!!!!!!!!!!!!!!!
		load-balance
		!!!!!!!!!!!!!!!! */
		if (rank == 0) {
			std::vector<size_t> total_partition_begin(num_bucket + 1, 0);

			mid_step_function("compute_collisions - com");
			MPI_Reduce(&partition_begin[1], &total_partition_begin[1],
				num_bucket, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator);
			mid_step_function("compute_collisions - prepare");

			iqs::utils::load_balancing_from_prefix_sum(total_partition_begin.begin(), total_partition_begin.end(),
				load_balancing_begin.begin(), load_balancing_begin.end());
		} else {
			mid_step_function("compute_collisions - com");
			MPI_Reduce(&partition_begin[1], NULL,
				num_bucket, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, communicator);
			mid_step_function("compute_collisions - prepare");
		}

		mid_step_function("compute_collisions - com");

		MPI_Bcast(&load_balancing_begin[1], n_segment, MPI_INT, 0, communicator);
		mid_step_function("compute_collisions - prepare");

		/* recompute local count and disp */
		local_disp[0] = 0;
		for (int i = 1; i <= n_segment; ++i) {
			local_disp[i] = partition_begin[load_balancing_begin[i]];
			local_count[i - 1] = local_disp[i] - local_disp[i - 1];
		}






		/* !!!!!!!!!!!!!!!!
		share
		!!!!!!!!!!!!!!!! */
		mid_step_function("compute_collisions - com");
		MPI_Alltoall(&local_count[0], num_threads, MPI_INT, &global_count[0], num_threads, MPI_INT, communicator);
		mid_step_function("compute_collisions - prepare");

		std::partial_sum(&global_count[0], &global_count[0] + n_segment, &global_disp[1]);

		/* recompute send and receive count and disp */
		send_disp[0] = 0; receive_count[0] = 0;
		for (int i = 1; i <= size; ++i) {
			send_disp[i] = local_disp[i*num_threads];
			send_count[i - 1] = send_disp[i] - send_disp[i - 1];

			receive_disp[i] = global_disp[i*num_threads];
			receive_count[i - 1] = receive_disp[i] - receive_disp[i - 1];
		}

		/* resize */
		buffer_resize(receive_disp[size]);

		/* actualy share partition */
		mid_step_function("compute_collisions - com");
		MPI_Alltoallv(&partitioned_hash[0], &send_count[0], &send_disp[0], MPI_UNSIGNED_LONG_LONG,
			&hash_buffer[0], &receive_count[0], &receive_disp[0], MPI_UNSIGNED_LONG_LONG, communicator);
		MPI_Alltoallv(&partitioned_mag[0], &send_count[0], &send_disp[0], mag_MPI_Datatype,
			&mag_buffer[0], &receive_count[0], &receive_disp[0], mag_MPI_Datatype, communicator);
		mid_step_function("compute_collisions - insert");






		/* !!!!!!!!!!!!!!!!
		compute-collision
		!!!!!!!!!!!!!!!! */
		/* prepare node_id buffer */
		for (int node = 0; node < size; ++node) {
			size_t begin = receive_disp[node], end = receive_disp[node + 1];
			#pragma omp parallel for
			for (size_t i = begin; i < end; ++i)
				node_id_buffer[i] = node;
		}

		#pragma omp parallel
		{
			std::vector<int> global_num_object_after_interferences(size, 0);

			int const thread_id = omp_get_thread_num();
			auto &elimination_map = elimination_maps[thread_id];

			/* compute total_size */
			size_t total_size = 0;
			for (int node_id = 0; node_id < size; ++node_id)
				total_size += global_count[node_id*num_threads + thread_id];
			elimination_map.reserve(total_size);

			/* insert into hashmap */
			for (int node_id = 0; node_id < size; ++node_id) {
				size_t begin = global_disp[node_id*num_threads + thread_id], end = global_disp[node_id*num_threads + thread_id + 1];
				for (size_t oid = begin; oid < end; ++oid) {

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
				}
			}
			elimination_map.clear();
		}






		
		/* !!!!!!!!!!!!!!!!
		share-back
		!!!!!!!!!!!!!!!! */
		mid_step_function("compute_collisions - com");
		MPI_Alltoallv(&mag_buffer[0], &receive_count[0], &receive_disp[0], mag_MPI_Datatype,
			&partitioned_mag[0], &send_count[0], &send_disp[0], mag_MPI_Datatype, communicator);
		MPI_Alltoallv(&is_unique_buffer[0], &receive_count[0], &receive_disp[0], MPI_CHAR,
			&partitioned_is_unique[0], &send_count[0], &send_disp[0], MPI_CHAR, communicator);
		mid_step_function("compute_collisions - finalize");

		/* un-partition magnitude */
		#pragma omp parallel for
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = next_oid[id];

			is_unique[oid] = partitioned_is_unique[id];
			magnitude[oid] = partitioned_mag[id];
		}






		
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
	distributed normalization function
	*/
	void mpi_iteration::normalize(MPI_Comm communicator, iqs::debug_t mid_step_function) {
		/* !!!!!!!!!!!!!!!!
		normalize
		 !!!!!!!!!!!!!!!! */
		mid_step_function("normalize");

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

		mid_step_function("end");
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
			size_t num_object_sent = num_symbolic_object_to_send / iqs::it_t::get_average_num_child();

			send_objects(num_object_sent, this_pair_id, communicator, true);
		} else if (num_symbolic_object < other_num_object)
			receive_objects(this_pair_id, communicator, true);
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