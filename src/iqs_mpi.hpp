#define PROBA_TYPE double

#include "iqs.hpp"

#include <mpi.h>

namespace iqs::mpi {
	namespace utils {        
		#include "utils/mpi_utils.hpp"
	}

	/* forward typedef */
	typedef class mpi_iteration mpi_it_t;
	typedef class mpi_symbolic_iteration mpi_sy_it_t;

	/*
	mpi iteration class
	*/
	class mpi_iteration : public iqs::iteration {
		friend mpi_symbolic_iteration;
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm comunicator, iqs::debug_t mid_step_function);

	protected:
		void normalize(MPI_Comm comunicator);

	public:
		mpi_iteration() {}
		mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

		void distribute_objects(MPI_Comm comunicator, int node_id);
		void gather_objects(MPI_Comm comunicator, int node_id);
	};

	class mpi_symbolic_iteration : public iqs::symbolic_iteration {
		friend mpi_iteration;
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm comunicator, iqs::debug_t mid_step_function); 

	protected:
		tbb::concurrent_hash_map<std::pair<size_t, uint32_t>, size_t> distributed_elimination_map;

		void compute_collisions(MPI_Comm comunicator);

	public:
		mpi_symbolic_iteration() {}
	};

	/*
	simulation function
	*/
	void simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm comunicator, iqs::debug_t mid_step_function=[](int){}) {
		MPI_Barrier(MPI_COMM_WORLD);

		iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
		symbolic_iteration.compute_collisions(comunicator);
		symbolic_iteration.finalize(rule, iteration, iteration_buffer, mid_step_function);
		iteration_buffer.normalize(comunicator);

		mid_step_function(8);
		
		std::swap(iteration_buffer, iteration);
	}

	/*
	distributed interference function
	*/
	void mpi_symbolic_iteration::compute_collisions(MPI_Comm comunicator) {
		/* Placeholder */
		iqs::symbolic_iteration::compute_collisions();

		/* TODO !!!! */
		int size, rank;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	}

	/*
	distributed normalization function
	*/
	void mpi_iteration::normalize(MPI_Comm comunicator) {
		int size, rank;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		/* !!!!!!!!!!!!!!!!
		step (8)
		 !!!!!!!!!!!!!!!! */

		PROBA_TYPE local_total_proba = 0;
		total_proba = 0;

		#pragma omp parallel for reduction(+:local_total_proba)
		for (size_t oid = 0; oid < num_object; ++oid) {
			PROBA_TYPE r = real[oid];
			PROBA_TYPE i = imag[oid];

			local_total_proba += r*r + i*i;
		}

		if (rank == 0) {
			/* add total proba for each node */
			total_proba = local_total_proba;
			for (int node = 1; node < size; ++node) {
				MPI_Recv(&local_total_proba, 1, MPI_DOUBLE, node, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
				total_proba += local_total_proba;
			}

			/* send back total proba */
			for (int node = 1; node < size; ++node)
				MPI_Send(&total_proba, 1, MPI_DOUBLE, node, 0 /* tag */, MPI_COMM_WORLD);
		} else {
			/* send local proba */
			MPI_Send(&local_total_proba, 1, MPI_DOUBLE, 0, 0 /* tag */, MPI_COMM_WORLD);

			/* receive total proba */
			MPI_Recv(&total_proba, 1, MPI_DOUBLE, 0, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
		}
		PROBA_TYPE normalization_factor = std::sqrt(total_proba);

		#pragma omp parallel for
		for (size_t oid = 0; oid < num_object; ++oid) {
			real[oid] /= normalization_factor;
			imag[oid] /= normalization_factor;
		}
	}

	/*
	function to distribute objects across nodes
	*/
	void mpi_iteration::distribute_objects(MPI_Comm comunicator, int node_id=0) {
		int size, rank;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		if (rank == node_id) {
			size_t begin_offset = 0;
			size_t begin_offset_0 = object_begin[num_object / size];
			for (int node = 1; node < size; ++node) {
				int node_to_send = node <= node_id ? node - 1 : node;

				/* compute sizes */
				size_t begin = (node * num_object) / size;
				size_t end = ((node + 1) * num_object) / size;
				size_t num_object_sent = end - begin;

				if (num_object_sent != 0) {
					/* take offset into account */
					begin_offset += object_begin[begin];

					#pragma omp parallel for
					for (size_t i = begin + 1; i <= end; ++i)
						object_begin[i] -= begin_offset;
				}
				
				/* send size */
				MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node_to_send, 0 /* tag */, MPI_COMM_WORLD);

				if (num_object_sent != 0) {
					/* send properties */
					MPI_Send(real.begin() + begin, num_object_sent, MPI_DOUBLE, node_to_send, 0 /* tag */, MPI_COMM_WORLD);
					MPI_Send(imag.begin() + begin, num_object_sent, MPI_DOUBLE, node_to_send, 0 /* tag */, MPI_COMM_WORLD);
					MPI_Send(object_begin.begin() + begin + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node_to_send, 0 /* tag */, MPI_COMM_WORLD);

					/* send objects */
					size_t objects_size = object_begin[end];
					MPI_Send(objects.begin() + begin_offset, objects_size, MPI_CHAR, node_to_send, 0 /* tag */, MPI_COMM_WORLD);
				}
			}

			/* resize this node */
			num_object = num_object / size;
			object_begin[num_object] = begin_offset_0;
			resize(num_object);
			objects.resize(object_begin[num_object]);
		} else {
			/* receive size */
			MPI_Recv(&num_object, 1, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);

			/* prepare state */
			resize(num_object);

			if (num_object != 0) {
				/* receive properties */
				MPI_Recv(real.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
				MPI_Recv(imag.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
				MPI_Recv(object_begin.begin() + 1, num_object, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);

				/* receive objects */
				size_t objects_size = object_begin[num_object];
				objects.zero_resize(objects_size);
				MPI_Recv(objects.begin(), objects_size, MPI_CHAR, node_id, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
			}
		}
	}

	/*
	function to gather object from all nodes
	*/
	void mpi_iteration::gather_objects(MPI_Comm comunicator, int node_id=0) {
		MPI_Barrier(MPI_COMM_WORLD);  
		
		int size, rank;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		if (rank == node_id) {
			for (int node = 1; node < size; ++node) {
				int receive_node = node <= node_id ? node - 1 : node;
				size_t next_num_object, num_object_sent;

				/* receive size */
				MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, receive_node, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
				if (num_object_sent != 0) {
					next_num_object = num_object_sent + num_object;

					/* prepare state */
					resize(next_num_object);

					/* receive properties */
					MPI_Recv(real.begin() + num_object, num_object_sent, MPI_DOUBLE, receive_node, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
					MPI_Recv(imag.begin() + num_object, num_object_sent, MPI_DOUBLE, receive_node, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);
					MPI_Recv(object_begin.begin() + num_object + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, receive_node, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);

					/* take offset into account */
					size_t objects_size = object_begin[next_num_object];
					size_t begin_offset = object_begin[num_object];

					#pragma omp parallel for
					for (size_t i = num_object + 1; i <= next_num_object; ++i)
						object_begin[i] += begin_offset;

					/* receive objects */
					objects.resize(objects_size);
					MPI_Recv(objects.begin() + begin_offset, objects_size, MPI_CHAR, receive_node, 0 /* tag */, MPI_COMM_WORLD, &utils::global_status);

					/* set size */
					num_object = next_num_object;
				}
			}

		} else {
			/* send size */
			MPI_Send(&num_object, 1, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, MPI_COMM_WORLD);

			if (num_object != 0) {
				/* send properties */
				MPI_Send(real.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, MPI_COMM_WORLD);
				MPI_Send(imag.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, MPI_COMM_WORLD);
				MPI_Send(object_begin.begin() + 1, num_object, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, MPI_COMM_WORLD);

				/* send objects */
				size_t objects_size = object_begin[num_object];
				MPI_Send(objects.begin(), objects_size, MPI_CHAR, node_id, 0 /* tag */, MPI_COMM_WORLD);

				/* clear this state */
				num_object = 0;
				resize(num_object);
				objects.zero_resize(0);
			}
		}
	}
}