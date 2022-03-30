#pragma once

#include "iqs.hpp"

#include <mpi.h>

#ifndef MIN_EQUALIZE_SIZE
	#define MIN_EQUALIZE_SIZE 100
#endif
#ifndef EQUALIZE_INBALANCE
	#define EQUALIZE_INBALANCE 0.05
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
	float equalize_inbalance = EQUALIZE_INBALANCE;
	#ifdef MINIMIZE_TRUNCATION
		bool minimize_truncation = true;
	#else
		bool minimize_truncation = false;
	#endif

	/* forward typedef */
	typedef class mpi_iteration mpi_it_t;
	typedef class mpi_symbolic_iteration mpi_sy_it_t;

	/*
	mpi iteration class
	*/
	class mpi_iteration : public iqs::iteration {
	private:
		friend mpi_symbolic_iteration;
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, iqs::debug_t mid_step_function);

		void equalize_symbolic(MPI_Comm communicator);
		void normalize(MPI_Comm communicator, iqs::debug_t mid_step_function=[](const char*){});



		/*
		utils functions
		*/
		size_t inline get_mem_size(MPI_Comm communicator) const {
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t)  + sizeof(float);

			size_t total_size, local_size = iteration_memory_size*magnitude.size() + objects.size();
			MPI_Allreduce(&local_size, &total_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_size;
		}
		float inline get_average_num_child(MPI_Comm communicator) const {
			return (float)get_total_truncated_num_child(communicator) / (float)get_total_truncated_num_object(communicator);
		}
		float inline get_average_object_size(MPI_Comm communicator) const {
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t)  + sizeof(float);

			size_t total_object_size;
			MPI_Allreduce(&object_begin[num_object], &total_object_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			size_t total_num_object = get_total_num_object(communicator);
			if (total_num_object == 0)
				return (float)iteration_memory_size;

			return (float)iteration_memory_size + (float)total_object_size/total_num_object;
		}
		size_t inline get_total_truncated_num_object(MPI_Comm communicator) const {
			size_t total_truncated_num_object;
			MPI_Allreduce(&truncated_num_object, &total_truncated_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_truncated_num_object;
		}
		size_t get_total_truncated_num_child(MPI_Comm communicator) const {
			size_t total_num_child = get_truncated_num_child();
			MPI_Allreduce(MPI_IN_PLACE, &total_num_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_num_child;
		}
		size_t get_total_num_child(MPI_Comm communicator) const {
			size_t total_num_child = get_num_symbolic_object();
			MPI_Allreduce(MPI_IN_PLACE, &total_num_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_num_child;
		}
		float get_average_num_child() {
			return iqs::iteration::get_average_num_child();
		}

		/*
		function to compute the maximum and minimum per node size
		*/
		float get_avg_num_symbolic_object_per_task(MPI_Comm communicator) {
			size_t total_num_object_per_node = get_num_symbolic_object();
			MPI_Allreduce(MPI_IN_PLACE, &total_num_object_per_node, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);

			int size;
			MPI_Comm_size(communicator, &size);

			return (float)total_num_object_per_node/size;
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
		PROBA_TYPE average_value(std::function<PROBA_TYPE(char const *object_begin, char const *object_end)> const observable) const {
			return node_total_proba == 0 ? 0 : iqs::iteration::average_value(observable) / node_total_proba * total_proba;
		}
		PROBA_TYPE average_value(std::function<PROBA_TYPE(char const *object_begin, char const *object_end)> const observable, MPI_Comm communicator) const {
			/* compute local average */
			PROBA_TYPE avg = iqs::iteration::average_value(observable);

			/* accumulate average value */
			MPI_Allreduce(MPI_IN_PLACE, &avg, 1, Proba_MPI_Datatype, MPI_SUM, communicator);
			return avg;
		}
		void send_objects(size_t num_object_sent, int node, MPI_Comm communicator, bool send_num_child=false) {
			const static size_t max_int = 1 << 31 - 1;

			/* send size */
			MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);
			if (num_object_sent == 0)
				return;
			size_t begin = num_object - num_object_sent;
			size_t send_object_size = object_begin[num_object] - object_begin[begin];
			MPI_Send(&send_object_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

			/* verify send */
			bool send;
			MPI_Recv(&send, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

			if (send) {
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
		void receive_objects(int node, MPI_Comm communicator, bool receive_num_child=false, size_t max_mem=-1) {
			const static size_t max_int = 1 << 31 - 1;

			/* receive size */
			size_t num_object_sent, send_object_size;
			MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);
			if (num_object_sent == 0)
				return;
			MPI_Recv(&send_object_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

			/* verify memory limit */
			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t)  + sizeof(float);
			bool recv = num_object_sent*iteration_memory_size + send_object_size < max_mem;
			MPI_Send(&recv, 1, MPI_CHAR, node, 0 /* tag*/, communicator);

			if (recv) {
				/* prepare state */
				size_t send_object_begin = object_begin[num_object];
				resize(num_object + num_object_sent);
				allocate(send_object_begin + send_object_size);

				/* receive properties */
				MPI_Recv(&magnitude[num_object], num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);
				MPI_Recv(&object_begin[num_object + 1], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

				/* receive objects */
				size_t object_offset = send_object_begin;
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

				if (receive_num_child) {
					/* receive num child */
					MPI_Recv(&num_childs[num_object], num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, MPI_STATUS_IGNORE);

					/* partial sum */
					num_childs[num_object] += child_begin[num_object];
					__gnu_parallel::partial_sum(num_childs.begin() + num_object, num_childs.begin() + num_object + num_object_sent, child_begin.begin() + num_object + 1);
					num_childs[num_object] -= child_begin[num_object];
				}

				num_object += num_object_sent;
			}
		}

		void equalize(MPI_Comm communicator);
		void distribute_objects(MPI_Comm communicator, int node_id);
		void gather_objects(MPI_Comm communicator, int node_id);

		size_t get_total_num_symbolic_object(MPI_Comm communicator) const {
			size_t total_num_child = get_num_symbolic_object();
			MPI_Allreduce(MPI_IN_PLACE, &total_num_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
			return total_num_child;
		}
	};

	class mpi_symbolic_iteration : public iqs::symbolic_iteration {
	private:
		friend mpi_iteration;
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &next_iteration, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, size_t max_num_object, iqs::debug_t mid_step_function); 

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

			static const size_t symbolic_iteration_memory_size = (1 + 1) + (2 + 2)*sizeof(PROBA_TYPE) + (5 + 1)*sizeof(size_t) + sizeof(uint32_t) + sizeof(float);

			static const size_t mpi_symbolic_iteration_memory_size = 1 + 2*sizeof(PROBA_TYPE) + 1*sizeof(size_t) + sizeof(int);
			return (float)symbolic_iteration_memory_size + (float)mpi_symbolic_iteration_memory_size + hash_map_size;
		}
		size_t inline get_mem_size(MPI_Comm communicator) const {
			static const size_t symbolic_iteration_memory_size = (1 + 1) + (2 + 2)*sizeof(PROBA_TYPE) + (5 + 1)*sizeof(size_t) + sizeof(uint32_t) + sizeof(float);
			size_t memory_size = magnitude.size()*symbolic_iteration_memory_size;

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

			static const size_t iteration_memory_size = 2*sizeof(PROBA_TYPE) + 4*sizeof(size_t)  + sizeof(float);

			size_t total_next_iteration_num_object = get_total_next_iteration_num_object(communicator);
			if (total_next_iteration_num_object == 0)
				return (float)iteration_memory_size;

			return (float)iteration_memory_size + (float)total_size/total_next_iteration_num_object;
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
		const int log_dimension = iqs::max_truncate_step == 0 && iqs::min_truncate_step == 0 ? 1/std::log(2) : 1/std::log(iqs::max_truncate_step) - 1/std::log(iqs::min_truncate_step);

		/* get local size */
		MPI_Comm localComm;
		int rank, size, local_size;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);
		MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_size(localComm, &local_size);



		


		if (size == 1)
			return iqs::simulate(iteration, rule, next_iteration, symbolic_iteration, max_num_object, mid_step_function);

		/* start actual simulation */
		iteration.compute_num_child(rule, mid_step_function);
		iteration.truncated_num_object = iteration.num_object;

		/* equalize symbolic objects */
		mid_step_function("equalize_child");
		for (int max_equalize = std::log2(size); max_equalize >= 0; --max_equalize) {
			/* check for condition */
			size_t max_n_object = iteration.get_max_num_object_per_task(communicator);
			size_t max_n_child = iteration.get_max_num_symbolic_object_per_task(communicator);
			float inbalance = ((float)max_n_child - iteration.get_avg_num_symbolic_object_per_task(communicator))/max_n_child;

			if (max_n_object < min_equalize_size || inbalance < equalize_inbalance)
				break;

			/* actually equalize */
			iteration.equalize_symbolic(communicator);
			iteration.truncated_num_object = iteration.num_object;
		}

		/* prepare truncate */
		mid_step_function("truncate_symbolic - prepare");
		iteration.prepare_truncate(mid_step_function);

		/* max_num_object */
		mid_step_function("truncate_symbolic");
		if (max_num_object == 0) {

			/* average object size with weighted average */
			float average_symbolic_object_size = symbolic_iteration.get_average_object_size(localComm);
			float average_object_size = iteration.get_average_object_size(localComm)*local_size;
			average_object_size += iteration.get_average_object_size(communicator);
			average_object_size /= local_size + 1;

			/* available memory */
			size_t avail_mem = next_iteration.get_mem_size(localComm);
			avail_mem += symbolic_iteration.get_mem_size(localComm);
			avail_mem += iqs::utils::get_free_mem();
			avail_mem *= 1 - iqs::safety_margin;

			/* actually truncate */
			size_t avg_truncate_symbolic_num_object, truncated_num_child = iteration.get_truncated_num_child();
			int max_truncate = iteration.num_object == 0 ? 0 : std::log2(size) + std::log(truncated_num_child)*log_dimension;
			MPI_Allreduce(MPI_IN_PLACE, &max_truncate, 1, MPI_INT, MPI_MAX, communicator);
			for (int i = max_truncate;; --i) {
				size_t total_num_child;
				MPI_Allreduce(&truncated_num_child, &total_num_child, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, localComm);
				size_t total_truncated_num_object = iteration.get_total_truncated_num_object(localComm);
				size_t used_memory = (total_truncated_num_object*average_object_size + total_num_child*average_symbolic_object_size)/iqs::utils::upsize_policy;



				/* !!!!!!
				!!!!!!!!!
				debuging */
				if (rank == 0)
					std::cerr << "\t" << (float)used_memory/1e9 << "GB =? " << (float)avail_mem/(1 - iqs::safety_margin)/1e9 << "GB * " << (1 - iqs::safety_margin) << ", " << i << "th iteration (symbolic truncation)\n";
				


				/* check for condition */
				if (i < 0 && used_memory <= avail_mem*(1 + iqs::truncation_tolerance))
					break;
				
				/* compute the number of child to keep with upword limit */
				size_t truncate_symbolic_num_object = used_memory == 0 ? 0 : total_num_child*avail_mem/used_memory/local_size;
				if (truncate_symbolic_num_object > truncated_num_child*iqs::max_truncate_step)
					truncate_symbolic_num_object = truncated_num_child*iqs::max_truncate_step;

				/* check for inbalance */
				if (!minimize_truncation) {
					if (i >= 0) {
						MPI_Allreduce(&truncate_symbolic_num_object, &avg_truncate_symbolic_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
						int count = iteration.num_object != 0;
						MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, communicator);
						if (count == 0)
							break;
						avg_truncate_symbolic_num_object /= count;
					}
					if (truncate_symbolic_num_object > avg_truncate_symbolic_num_object)
						truncate_symbolic_num_object = (truncate_symbolic_num_object + avg_truncate_symbolic_num_object)/2;
				}

				/* set limits on the number of childs */
				if (truncate_symbolic_num_object < truncated_num_child*iqs::min_truncate_step)
					truncate_symbolic_num_object = truncated_num_child*iqs::min_truncate_step;
				if (i < 0 && truncate_symbolic_num_object > truncated_num_child)
					truncate_symbolic_num_object = truncated_num_child;

				/* actually truncate */
				iteration.truncate_symbolic(truncate_symbolic_num_object, mid_step_function);
				truncated_num_child = truncate_symbolic_num_object;
			}
		} else
			iteration.truncate(max_num_object/local_size, mid_step_function);

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



		/* prepare truncate */
		mid_step_function("truncate - prepare");
		symbolic_iteration.prepare_truncate(mid_step_function);

		/* second max_num_object */
		mid_step_function("truncate");
		if (max_num_object == 0) {

			/* available memory */
			size_t avail_mem = next_iteration.get_mem_size(localComm);
			avail_mem += iqs::utils::get_free_mem();
			avail_mem *= 1 - iqs::safety_margin;

			/* actually truncate */
			size_t avg_truncate_num_object;
			int max_truncate = symbolic_iteration.num_object_after_interferences == 0 ? 0 : std::log2(size) + std::log(symbolic_iteration.num_object_after_interferences)*log_dimension;
			MPI_Allreduce(MPI_IN_PLACE, &max_truncate, 1, MPI_INT, MPI_MAX, communicator);
			for (int i = max_truncate;; --i) {
				size_t total_num_child = symbolic_iteration.get_total_next_iteration_num_object(localComm);
				size_t used_memory = symbolic_iteration.get_average_child_size(localComm)*total_num_child/iqs::utils::upsize_policy;


				/* !!!!!!
				!!!!!!!!!
				debuging */
				if (rank == 0)
					std::cerr << "\t" << (float)used_memory/1e9 << "GB =? " << (float)avail_mem/(1 - iqs::safety_margin)/1e9 << "GB * " << (1 - iqs::safety_margin) << ", " << i << "th iteration (truncation)\n";
				


				/* check for condition */
				if (i < 0 && used_memory <= avail_mem*(1 + iqs::truncation_tolerance))
					break;

				/* compute the number of child to keep with upword limit */
				size_t truncate_num_object = used_memory == 0 ? 0 : total_num_child*avail_mem/used_memory/local_size;
				if (truncate_num_object > total_num_child*iqs::max_truncate_step)
					truncate_num_object = total_num_child*iqs::max_truncate_step;

				/* check for inbalance */
				if (!minimize_truncation) {
					if (i >= 0) {
						MPI_Allreduce(&truncate_num_object, &avg_truncate_num_object, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, communicator);
						int count = symbolic_iteration.num_object_after_interferences != 0;
						MPI_Allreduce(MPI_IN_PLACE, &count, 1, MPI_INT, MPI_SUM, communicator);
						if (count == 0)
							break; 
						avg_truncate_num_object /= count;
					}
					if (truncate_num_object > avg_truncate_num_object)
						truncate_num_object = (truncate_num_object + avg_truncate_num_object) / 2;
				}

				/* set limits on the number of childs */
				if (truncate_num_object < symbolic_iteration.next_iteration_num_object*iqs::min_truncate_step)
					truncate_num_object = symbolic_iteration.next_iteration_num_object*iqs::min_truncate_step;
				if (i < 0 && truncate_num_object > symbolic_iteration.next_iteration_num_object)
					truncate_num_object = (1 - iqs::truncation_tolerance)*symbolic_iteration.next_iteration_num_object;

				/* truncate */
				if (!(symbolic_iteration.next_iteration_num_object < truncate_num_object*(1 + iqs::truncation_tolerance) &&
					symbolic_iteration.next_iteration_num_object > truncate_num_object*(1 - iqs::truncation_tolerance)))
						symbolic_iteration.truncate(truncate_num_object, mid_step_function);
			}
		} else
			symbolic_iteration.truncate(max_num_object/local_size, mid_step_function);

		/* finalize simulation */
		symbolic_iteration.finalize(rule, iteration, next_iteration, mid_step_function);
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

		/* recompute local count and disp */
		mid_step_function("compute_collisions - prepare");
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






		/* !!!!!!!!!!!!!!!!
		compute-collision
		!!!!!!!!!!!!!!!! */
		mid_step_function("compute_collisions - prepare");
		/* prepare node_id buffer */
		for (int node = 0; node < size; ++node) {
			size_t begin = receive_disp[node], end = receive_disp[node + 1];
			#pragma omp parallel for
			for (size_t i = begin; i < end; ++i)
				node_id_buffer[i] = node;
		}

		mid_step_function("compute_collisions - insert");
		#pragma omp parallel
		{
			std::vector<int> global_num_object_after_interferences(size, 0);

			int const thread_id = omp_get_thread_num();
			robin_hood::unordered_map<size_t, size_t> elimination_map;

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
		}






		
		/* !!!!!!!!!!!!!!!!
		share-back
		!!!!!!!!!!!!!!!! */
		mid_step_function("compute_collisions - com");
		MPI_Alltoallv(&mag_buffer[0], &receive_count[0], &receive_disp[0], mag_MPI_Datatype,
			&partitioned_mag[0], &send_count[0], &send_disp[0], mag_MPI_Datatype, communicator);
		MPI_Alltoallv(&is_unique_buffer[0], &receive_count[0], &receive_disp[0], MPI_CHAR,
			&partitioned_is_unique[0], &send_count[0], &send_disp[0], MPI_CHAR, communicator);

		/* un-partition magnitude */
		mid_step_function("compute_collisions - finalize");
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

		MPI_Comm localComm;
		int rank, size, local_size;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);
		MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_size(localComm, &local_size);

		int this_pair_id;
		if (rank == 0) {
			/* gather sizes */
			std::vector<size_t> sizes(size, 0);
			MPI_Gather(&num_object, 1, MPI_UNSIGNED_LONG_LONG, &sizes[0], 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);

			/* compute pair_id*/
			std::vector<int> pair_id(size, 0);
			utils::make_equal_pairs(&sizes[0], &sizes[0] + size, &pair_id[0]);

			/* scatter pair_id */
			MPI_Scatter(&pair_id[0], 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
		} else {
			/* gather sizes */
			MPI_Gather(&num_object, 1, MPI_UNSIGNED_LONG_LONG, NULL, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);

			/* scatter pair_id */
			MPI_Scatter(NULL, 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
		}

		/* get available memory */
		MPI_Barrier(localComm);
		size_t total_iteration_size, iteration_size = iqs::iteration::get_mem_size();
		MPI_Allreduce(&iteration_size, &total_iteration_size, 1, Proba_MPI_Datatype, MPI_SUM, localComm);
		size_t avail_mem = ((iqs::utils::get_free_mem() + total_iteration_size)/local_size - iteration_size)*(1 - iqs::safety_margin);
		MPI_Barrier(localComm);

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
			send_objects(num_object_sent, this_pair_id, communicator, false);
		} else if (num_object < other_num_object)
			receive_objects(this_pair_id, communicator, false, avail_mem);
	}

	/*
	equalize symbolic object across nodes
	*/
	void mpi_iteration::equalize_symbolic(MPI_Comm communicator) {
		MPI_Request request = MPI_REQUEST_NULL;

		MPI_Comm localComm;
		int rank, size, local_size;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);
		MPI_Comm_split_type(communicator, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm);
		MPI_Comm_size(localComm, &local_size);

		/* compute the number of symbolic objects */
		size_t num_symbolic_object = child_begin[num_object];

		int this_pair_id;
		if (rank == 0) {
			/* gather sizes */
			std::vector<size_t> sizes(size, 0);
			MPI_Gather(&num_symbolic_object, 1, MPI_UNSIGNED_LONG_LONG, &sizes[0], 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);

			/* compute pair_id*/
			std::vector<int> pair_id(size, 0);
			utils::make_equal_pairs(&sizes[0], &sizes[0] + size, &pair_id[0]);

			/* scatter pair_id */
			MPI_Scatter(&pair_id[0], 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
		} else {
			/* gather sizes */
			MPI_Gather(&num_symbolic_object, 1, MPI_UNSIGNED_LONG_LONG, NULL, 1, MPI_UNSIGNED_LONG_LONG, 0, communicator);

			/* scatter pair_id */
			MPI_Scatter(NULL, 1, MPI_INT, &this_pair_id, 1, MPI_INT, 0, communicator);
		}
		
		/* get available memory */
		MPI_Barrier(localComm);
		size_t total_iteration_size, iteration_size = iqs::iteration::get_mem_size();
		MPI_Allreduce(&iteration_size, &total_iteration_size, 1, Proba_MPI_Datatype, MPI_SUM, localComm);
		size_t avail_mem = ((iqs::utils::get_free_mem() + total_iteration_size)/local_size - iteration_size)*(1 - iqs::safety_margin);
		MPI_Barrier(localComm);

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

			/* find the actual number of object to send */
			auto limit_it = std::lower_bound(child_begin.begin(), child_begin.begin() + num_object, num_symbolic_object - num_symbolic_object_to_send);
			size_t num_object_sent = std::distance(limit_it, child_begin.begin() + num_object);

			send_objects(num_object_sent, this_pair_id, communicator, true);
		} else if (num_symbolic_object < other_num_object)
			receive_objects(this_pair_id, communicator, true, avail_mem);
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