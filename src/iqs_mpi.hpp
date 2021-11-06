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
	function to distribute objects across nodes
	*/
	void mpi_iteration::distribute_objects(MPI_Comm comunicator, int node_id=0) {
		int size, rank;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		if (rank == node_id) {

		} else {

		}
	}

	/*
	function to gather object from all nodes
	*/
	void mpi_iteration::gather_objects(MPI_Comm comunicator, int node_id=0) {
		int size, rank;
		MPI_Comm_size(MPI_COMM_WORLD, &size);
		MPI_Comm_rank(MPI_COMM_WORLD, &rank);

		if (rank == node_id) {

		} else {

		}
	}

	/*
	simulation function
	*/
	void simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm comunicator, iqs::debug_t mid_step_function=[](int){}) {
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
}