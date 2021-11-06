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
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, iqs::debug_t mid_step_function);

	protected:
		void normalize(MPI_Comm communicator);

	public:
		mpi_iteration() {}
		mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

		void distribute_objects(MPI_Comm communicator, int node_id);
		void gather_objects(MPI_Comm communicator, int node_id);
	};

	class mpi_symbolic_iteration : public iqs::symbolic_iteration {
		friend mpi_iteration;
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, iqs::debug_t mid_step_function); 

	protected:
		tbb::concurrent_hash_map<size_t, std::pair<size_t, int>> distributed_elimination_map;

		iqs::utils::numa_vector<PROBA_TYPE> partitioned_real, partitioned_imag;
		iqs::utils::numa_vector<size_t> partitioned_hash;
		iqs::utils::numa_vector<size_t> original_id;
		iqs::utils::numa_vector<bool> partitioned_is_unique;

		struct node_map_type {
			size_t num_object = 0;
			size_t num_object_after_interferences = 0;

			iqs::utils::numa_vector<PROBA_TYPE> real, imag;
			iqs::utils::numa_vector<size_t> hash;
			iqs::utils::numa_vector<bool> is_unique;

			void resize(size_t size) {
				real.resize(size);
				imag.resize(size);
				hash.resize(size);
				is_unique.resize(size);
			}
		};
		std::vector<node_map_type> node_map;

		void compute_collisions(MPI_Comm communicator);
		void resize(size_t size) {
			iqs::symbolic_iteration::resize(size);

			partitioned_real.resize(size);
			partitioned_imag.resize(size);
			partitioned_hash.resize(size);
			partitioned_is_unique.resize(size);
		}

		long long int memory_size = (1 + 2) + (2 + 4)*sizeof(PROBA_TYPE) + (6 + 4)*sizeof(size_t) + sizeof(uint32_t) + sizeof(double);

	public:
		mpi_symbolic_iteration() {}
	};

	/*
	simulation function
	*/
	void simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, iqs::debug_t mid_step_function=[](int){}) {
		iteration.generate_symbolic_iteration(rule, symbolic_iteration, mid_step_function);
		symbolic_iteration.compute_collisions(communicator);
		symbolic_iteration.finalize(rule, iteration, iteration_buffer, mid_step_function);
		iteration_buffer.normalize(communicator);

		mid_step_function(8);
		
		std::swap(iteration_buffer, iteration);
	}

	/*
	distributed interference function
	*/
	void mpi_symbolic_iteration::compute_collisions(MPI_Comm communicator) {
		/*
		function to add a key
		*/
		auto static const insert_key = [&](size_t oid, int node_id) {
			size_t hash = node_map[node_id].hash[oid];

			/* accessing key */
			tbb::concurrent_hash_map<size_t, std::pair<size_t, int>>::accessor it;
			if (distributed_elimination_map.insert(it, {hash, {oid, node_id}})) {
				node_map[node_id].is_unique[oid] = true; /* keep this graph */
			} else {
				auto [other_id, other_node_id] = it->second;

				bool is_greater = node_map[node_id].num_object_after_interferences > node_map[other_node_id].num_object_after_interferences;
				if (is_greater) {
					/* if it exist add the probabilities */
					node_map[other_node_id].real[other_id] += node_map[node_id].real[oid];
					node_map[other_node_id].imag[other_id] += node_map[node_id].imag[oid];

					/* discard this graph */
					node_map[node_id].is_unique[oid] = false;
				} else {
					/* if the size aren't balanced, add the probabilities */
					node_map[node_id].real[oid] += node_map[other_node_id].real[other_id];
					node_map[node_id].imag[oid] += node_map[other_node_id].imag[other_id];

					/* discard the other graph */
					node_map[node_id].is_unique[oid] = true;
					node_map[other_node_id].is_unique[other_id] = false;
				}
			}
		};

		/* !!!!!!!!!!!!!!!!
		step (4)

		Actual code :
		 !!!!!!!!!!!!!!!! */

		MPI_Barrier(communicator);

		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		node_map.resize(size);

		/* partition nodes */
		int *modulo_begin = (int*)calloc(size + 1, sizeof(int));
		utils::generalized_modulo_partition(next_oid.begin(), next_oid.begin() + num_object,
			hash.begin(), modulo_begin,
			size);

		/* generate partitioned hash */
		#pragma omp parallel for schedule(static)
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = next_oid[id];

			partitioned_real[id] = real[oid];
			partitioned_imag[id] = imag[oid];
			partitioned_hash[id] = hash[oid];
			original_id[oid] = id;
		}

		/* share partitions */
		for (int node = 0; node < size; ++node)
			if (rank == node) {
				for (int receive_node = 0; receive_node < size; ++receive_node)
					if (receive_node == rank) {

						/* resize */
						size_t this_size = modulo_begin[node + 1] - modulo_begin[node];
						node_map[receive_node].num_object = this_size;
						node_map[receive_node].num_object_after_interferences = 0;
						node_map[receive_node].resize(this_size);
						
						/* copy partitioned properties into node map */
						#pragma omp parallel for schedule(static)
						for (size_t i = 0; i < this_size; ++i) {
							size_t id = modulo_begin[receive_node] + i;

							node_map[receive_node].real[i] = partitioned_real[id];
							node_map[receive_node].imag[i] = partitioned_imag[id];
							node_map[receive_node].hash[i] = partitioned_hash[id];
						}
					} else {
						/* receive size */
						size_t this_size;
						MPI_Recv(&this_size, 1, MPI_UNSIGNED_LONG_LONG, receive_node, 0 /* tag */, communicator, &utils::global_status);

						/* prepare receive */
						node_map[receive_node].num_object = this_size;
						node_map[receive_node].num_object_after_interferences = 0;
						node_map[receive_node].resize(this_size);

						/* receive properties */
						if (this_size > 0) {
							MPI_Recv(node_map[receive_node].real.begin(), this_size, MPI_DOUBLE, receive_node, 0 /* tag */, communicator, &utils::global_status);
							MPI_Recv(node_map[receive_node].imag.begin(), this_size, MPI_DOUBLE, receive_node, 0 /* tag */, communicator, &utils::global_status);
							MPI_Recv(node_map[receive_node].hash.begin(), this_size, MPI_UNSIGNED_LONG_LONG, receive_node, 0 /* tag */, communicator, &utils::global_status);
						}
					}
			} else {
				/* send size */
				size_t this_size = modulo_begin[node + 1] - modulo_begin[node];
				MPI_Send(&this_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

				/* send properties */
				if (this_size > 0) {
					MPI_Send(partitioned_real.begin() + modulo_begin[node], this_size, MPI_DOUBLE, node, 0 /* tag */, communicator);
					MPI_Send(partitioned_imag.begin() + modulo_begin[node], this_size, MPI_DOUBLE, node, 0 /* tag */, communicator);
					MPI_Send(partitioned_hash.begin() + modulo_begin[node], this_size, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);
				}
			}

		/* generate the interference table */
		for (int node = 0; node < size; ++node) {
			bool fast = false;
			bool skip_test = node_map[node].num_object < iqs::utils::min_vector_size;
			size_t test_size = skip_test ? 0 : node_map[node].num_object*collision_test_proportion;

			/* first fast test */
			if (!skip_test) {
				#pragma omp parallel for schedule(static)
				for (size_t oid = 0; oid < test_size; ++oid) //size_t oid = oid[i];
					insert_key(oid, node);

				fast = test_size - elimination_map.size() < test_size*collision_test_proportion;
				if (fast)
					for (size_t oid = test_size; oid < node_map[node].num_object; ++oid)
						node_map[node].is_unique[oid] = true;
			}

			/* second complete test */
			if (!fast)
				#pragma omp parallel for schedule(static)
				for (size_t oid = test_size; oid < node_map[node].num_object; ++oid) //size_t oid = oid[i];
					insert_key(oid, node);
		}

		/* share back partitions */
		for (int node = 0; node < size; ++node)
			if (rank == node) {
				for (int receive_node = 0; receive_node < size; ++receive_node)
					if (receive_node == rank) {

						/* copy local data */
						size_t this_size = modulo_begin[receive_node + 1] - modulo_begin[receive_node];
						#pragma omp for schedule(static)
						for (size_t i = 0; i < this_size; ++i) {
							size_t id = modulo_begin[receive_node] + i;

							partitioned_real[id] = node_map[receive_node].real[i];
							partitioned_imag[id] = node_map[receive_node].imag[i];
							partitioned_is_unique[id] = node_map[receive_node].is_unique[i];
						}
					} else {

						/* receive properties */
						size_t this_size = modulo_begin[receive_node + 1] - modulo_begin[receive_node];
						if (this_size > 0) {
							MPI_Recv(partitioned_real.begin() + modulo_begin[receive_node], this_size, MPI_DOUBLE, receive_node, 0 /* tag */, communicator, &utils::global_status);
							MPI_Recv(partitioned_imag.begin() + modulo_begin[receive_node], this_size, MPI_DOUBLE, receive_node, 0 /* tag */, communicator, &utils::global_status);
							MPI_Recv(partitioned_is_unique.begin() + modulo_begin[receive_node], this_size, MPI_CHAR, receive_node, 0 /* tag */, communicator, &utils::global_status);
						}
					}
			} else {

				/* send properties */
				size_t this_size = node_map[node].num_object;
				if (this_size > 0) {
					MPI_Send(node_map[node].real.begin(), this_size, MPI_DOUBLE, node, 0 /* tag */, communicator);
					MPI_Send(node_map[node].imag.begin(), this_size, MPI_DOUBLE, node, 0 /* tag */, communicator);
					MPI_Send(node_map[node].is_unique.begin(), this_size, MPI_CHAR, node, 0 /* tag */, communicator);
				}
			}

		/* regenerate real, imag and is_unique */
		#pragma omp parallel for schedule(static)
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = original_id[id];

			real[oid] = partitioned_real[id];
			imag[oid] = partitioned_imag[id];
			is_unique[oid] = partitioned_is_unique[id];
		}

		/* keep only unique objects */
		auto partitioned_it = __gnu_parallel::partition(next_oid.begin(), next_oid.begin() + num_object,
			[&](size_t const oid) {
				/* check if graph is unique */
				if (!is_unique[oid])
					return false;

				/* check for zero probability */
				PROBA_TYPE r = real[oid];
				PROBA_TYPE i = imag[oid];

				return r*r + i*i > iqs::tolerance;
			});
		num_object_after_interferences = std::distance(next_oid.begin(), partitioned_it);

		/* clear map */
		distributed_elimination_map.clear();
	}

	/*
	distributed normalization function
	*/
	void mpi_iteration::normalize(MPI_Comm communicator) {
		MPI_Barrier(communicator);

		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

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

		/* accumulate probabilities on the master node */
		if (rank == 0) {
			/* add total proba for each node */
			total_proba = local_total_proba;
			for (int node = 1; node < size; ++node) {
				MPI_Recv(&local_total_proba, 1, MPI_DOUBLE, node, 0 /* tag */, communicator, &utils::global_status);
				total_proba += local_total_proba;
			}

			/* send back total proba */
			for (int node = 1; node < size; ++node)
				MPI_Send(&total_proba, 1, MPI_DOUBLE, node, 0 /* tag */, communicator);
		} else {
			/* send local proba */
			MPI_Send(&local_total_proba, 1, MPI_DOUBLE, 0, 0 /* tag */, communicator);

			/* receive total proba */
			MPI_Recv(&total_proba, 1, MPI_DOUBLE, 0, 0 /* tag */, communicator, &utils::global_status);
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
	void mpi_iteration::distribute_objects(MPI_Comm communicator, int node_id=0) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

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
				MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node_to_send, 0 /* tag */, communicator);

				if (num_object_sent != 0) {
					/* send properties */
					MPI_Send(real.begin() + begin, num_object_sent, MPI_DOUBLE, node_to_send, 0 /* tag */, communicator);
					MPI_Send(imag.begin() + begin, num_object_sent, MPI_DOUBLE, node_to_send, 0 /* tag */, communicator);
					MPI_Send(object_begin.begin() + begin + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node_to_send, 0 /* tag */, communicator);

					/* send objects */
					size_t objects_size = object_begin[end];
					MPI_Send(objects.begin() + begin_offset, objects_size, MPI_CHAR, node_to_send, 0 /* tag */, communicator);
				}
			}

			/* resize this node */
			num_object = num_object / size;
			object_begin[num_object] = begin_offset_0;
			resize(num_object);
			objects.resize(object_begin[num_object]);
		} else {
			/* receive size */
			MPI_Recv(&num_object, 1, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, communicator, &utils::global_status);

			/* prepare state */
			resize(num_object);

			if (num_object != 0) {
				/* receive properties */
				MPI_Recv(real.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, communicator, &utils::global_status);
				MPI_Recv(imag.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, communicator, &utils::global_status);
				MPI_Recv(object_begin.begin() + 1, num_object, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, communicator, &utils::global_status);

				/* receive objects */
				size_t objects_size = object_begin[num_object];
				objects.zero_resize(objects_size);
				MPI_Recv(objects.begin(), objects_size, MPI_CHAR, node_id, 0 /* tag */, communicator, &utils::global_status);
			}
		}
	}

	/*
	function to gather object from all nodes
	*/
	void mpi_iteration::gather_objects(MPI_Comm communicator, int node_id=0) {
		MPI_Barrier(communicator);  

		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		if (rank == node_id) {
			for (int node = 1; node < size; ++node) {
				int receive_node = node <= node_id ? node - 1 : node;
				size_t next_num_object, num_object_sent;

				/* receive size */
				MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, receive_node, 0 /* tag */, communicator, &utils::global_status);
				if (num_object_sent != 0) {
					next_num_object = num_object_sent + num_object;

					/* prepare state */
					resize(next_num_object);

					/* receive properties */
					MPI_Recv(real.begin() + num_object, num_object_sent, MPI_DOUBLE, receive_node, 0 /* tag */, communicator, &utils::global_status);
					MPI_Recv(imag.begin() + num_object, num_object_sent, MPI_DOUBLE, receive_node, 0 /* tag */, communicator, &utils::global_status);
					MPI_Recv(object_begin.begin() + num_object + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, receive_node, 0 /* tag */, communicator, &utils::global_status);

					/* take offset into account */
					size_t objects_size = object_begin[next_num_object];
					size_t begin_offset = object_begin[num_object];

					#pragma omp parallel for
					for (size_t i = num_object + 1; i <= next_num_object; ++i)
						object_begin[i] += begin_offset;

					/* receive objects */
					objects.resize(objects_size);
					MPI_Recv(objects.begin() + begin_offset, objects_size, MPI_CHAR, receive_node, 0 /* tag */, communicator, &utils::global_status);

					/* set size */
					num_object = next_num_object;
				}
			}

		} else {
			/* send size */
			MPI_Send(&num_object, 1, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, communicator);

			if (num_object != 0) {
				/* send properties */
				MPI_Send(real.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, communicator);
				MPI_Send(imag.begin(), num_object, MPI_DOUBLE, node_id, 0 /* tag */, communicator);
				MPI_Send(object_begin.begin() + 1, num_object, MPI_UNSIGNED_LONG_LONG, node_id, 0 /* tag */, communicator);

				/* send objects */
				size_t objects_size = object_begin[num_object];
				MPI_Send(objects.begin(), objects_size, MPI_CHAR, node_id, 0 /* tag */, communicator);

				/* clear this state */
				num_object = 0;
				resize(num_object);
				objects.zero_resize(0);
			}
		}
	}
}