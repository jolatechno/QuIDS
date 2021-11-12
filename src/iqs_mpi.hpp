#include "iqs.hpp"

#include <mpi.h>

namespace iqs::mpi {
	namespace utils {        
		#include "utils/mpi_utils.hpp"
	}

	/* mpi auto type */
	const static MPI_Datatype Proba_MPI_Datatype = utils::get_mpi_datatype((PROBA_TYPE)0);
	const static MPI_Datatype mag_MPI_Datatype = utils::get_mpi_datatype((std::complex<PROBA_TYPE>)0);

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
		PROBA_TYPE node_total_proba = 0;

		mpi_iteration() {}
		mpi_iteration(char* object_begin_, char* object_end_) : iqs::iteration(object_begin_, object_end_) {}

		size_t get_total_num_object(MPI_Comm communicator) {
			int size, rank;
			MPI_Comm_size(communicator, &size);
			MPI_Comm_rank(communicator, &rank);

			/* accumulate number of node */
			size_t total_num_object = num_object;
			if (rank == 0) {
				/* add the number of node for each node */
				size_t local_num_object;
				for (int node = 1; node < size; ++node) {
					MPI_Recv(&local_num_object, 1, MPI::UNSIGNED_LONG, node, 0 /* tag */, communicator, &utils::global_status);
					total_num_object += local_num_object;
				}

				/* send back number of node */
				for (int node = 1; node < size; ++node)
					MPI_Send(&total_num_object, 1, MPI::UNSIGNED_LONG, node, 0 /* tag */, communicator);
			} else {
				/* send local number of node */
				MPI_Send(&num_object, 1, MPI::UNSIGNED_LONG, 0, 0 /* tag */, communicator);

				/* receive total number of node */
				MPI_Recv(&total_num_object, 1, MPI::UNSIGNED_LONG, 0, 0 /* tag */, communicator, &utils::global_status);
			}

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
			T avg = iqs::iteration::average_value(observable);

			/* accumulate average value */
			MPI_Datatype avg_datatype = utils::get_mpi_datatype(avg);
			if (rank == 0) {
				/* add avgerages for each node */
				T local_avg;
				for (int node = 1; node < size; ++node) {
					MPI_Recv(&local_avg, 1, avg_datatype, node, 0 /* tag */, communicator, &utils::global_status);
					avg += local_avg;
				}

				/* send back total avgerage */
				for (int node = 1; node < size; ++node)
					MPI_Send(&avg, 1, avg_datatype, node, 0 /* tag */, communicator);
			} else {
				/* send local avgerage */
				MPI_Send(&avg, 1, avg_datatype, 0, 0 /* tag */, communicator);

				/* receive total avgerage */
				MPI_Recv(&avg, 1, avg_datatype, 0, 0 /* tag */, communicator, &utils::global_status);
			}

			return avg;
		}
		void send_objects(size_t num_object_sent, int node, MPI_Comm communicator) {
			/* send size */
			MPI_Send(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

			if (num_object_sent != 0) {
				size_t begin = num_object - num_object_sent;

				/* prepare send */
				size_t send_object_begin = object_begin[begin];
				#pragma omp parallel for schedule(static)
				for (size_t i = begin + 1; i <= num_object; ++i)
					object_begin[i] -= send_object_begin;

				/* send properties */
				MPI_Send(magnitude.begin() + begin, num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator);
				MPI_Send(object_begin.begin() + begin + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

				/* send objects */
				size_t send_object_size = object_begin[num_object];
				MPI_Send(objects.begin() + send_object_begin, send_object_size, MPI_CHAR, node, 0 /* tag */, communicator);

				/* pop */
				pop(num_object_sent, false);
			}
		}
		void receive_objects(int node, MPI_Comm communicator) {
			/* receive size */
			size_t num_object_sent;
			MPI_Recv(&num_object_sent, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, &utils::global_status);

			if (num_object_sent != 0) {
				/* prepare state */
				resize(num_object + num_object_sent);

				/* receive properties */
				MPI_Recv(magnitude.begin() + num_object, num_object_sent, mag_MPI_Datatype, node, 0 /* tag */, communicator, &utils::global_status);
				MPI_Recv(object_begin.begin() + num_object + 1, num_object_sent, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, &utils::global_status);

				/* prepare receive objects */
				size_t send_object_begin = object_begin[num_object];
				size_t send_object_size = object_begin[num_object + num_object_sent];
				allocate(send_object_begin + send_object_size);

				/* receive objects */
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
		friend void inline simulate(mpi_it_t &iteration, iqs::rule_t const *rule, mpi_it_t &iteration_buffer, mpi_sy_it_t &symbolic_iteration, MPI_Comm communicator, iqs::debug_t mid_step_function); 

	protected:
		tbb::concurrent_hash_map<size_t, std::pair<size_t, int>> distributed_elimination_map;

		iqs::utils::numa_vector<mag_t> partitioned_mag;
		iqs::utils::numa_vector<size_t> partitioned_hash;
		iqs::utils::numa_vector<bool> partitioned_is_unique;

		struct node_map_type {
			size_t num_object = 0;
			size_t num_object_after_interferences = 0;

			iqs::utils::numa_vector<mag_t> magnitude;
			iqs::utils::numa_vector<size_t> hash;
			iqs::utils::numa_vector<bool> is_unique;

			void resize(size_t size) {
				magnitude.resize(size);
				hash.resize(size);
				is_unique.resize(size);
			}
		};
		std::vector<node_map_type> node_map;

		void compute_collisions(MPI_Comm communicator);
		void resize(size_t size) {
			iqs::symbolic_iteration::resize(size);

			partitioned_mag.resize(size);
			partitioned_hash.resize(size);
			partitioned_is_unique.resize(size);
		}

		long long int memory_size = (1 + 3) + (2 + 4)*sizeof(PROBA_TYPE) + (6 + 4)*sizeof(size_t) + sizeof(uint32_t) + sizeof(double) + sizeof(int);

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
		int *modulo_begin;
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
		auto static const insert_key = [&](size_t oid, int node_id) {
			size_t hash = node_map[node_id].hash[oid];

			/* accessing key */
			tbb::concurrent_hash_map<size_t, std::pair<size_t, int>>::accessor it;
			bool unique = distributed_elimination_map.insert(it, {hash, {oid, node_id}});
			if (unique) {
				/* keep this graph */
				node_map[node_id].is_unique[oid] = true;

				/* increment values */
				#pragma omp atomic 
				++node_map[node_id].num_object_after_interferences;
			} else {
				auto [other_id, other_node_id] = it->second;

				bool is_greater = node_map[node_id].num_object_after_interferences >= node_map[other_node_id].num_object_after_interferences;
				if (is_greater) {
					/* if it exist add the probabilities */
					node_map[other_node_id].magnitude[other_id] += node_map[node_id].magnitude[oid];

					/* discard this graph */
					node_map[node_id].is_unique[oid] = false;
				} else {
					/* if the size aren't balanced, add the probabilities */
					node_map[node_id].magnitude[oid] += node_map[other_node_id].magnitude[other_id];

					/* discard the other graph */
					node_map[node_id].is_unique[oid] = true;
					node_map[other_node_id].is_unique[other_id] = false;

					/* increment values */
					#pragma omp atomic 
					++node_map[node_id].num_object_after_interferences;
					#pragma omp atomic 
					--node_map[other_node_id].num_object_after_interferences;
				}
			}
		};

		/*
		function to receive initial data
		*/
		static const auto receive_data = [&](int node) {
			/* receive size */
			size_t this_size;
			MPI_Recv(&this_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, &utils::global_status);

			/* prepare receive */
			node_map[node].num_object = this_size;
			node_map[node].num_object_after_interferences = 0;
			node_map[node].resize(this_size);

			/* receive properties */
			if (this_size > 0) {
				MPI_Recv(node_map[node].magnitude.begin(), this_size, mag_MPI_Datatype, node, 0 /* tag */, communicator, &utils::global_status);
				MPI_Recv(node_map[node].hash.begin(), this_size, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator, &utils::global_status);
			}
		};

		/*
		function to send initial data
		*/
		static const auto send_data = [&](int node) {
			/* send size */
			size_t this_size = modulo_begin[node + 1] - modulo_begin[node];
			MPI_Send(&this_size, 1, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);

			/* send properties */
			if (this_size > 0) {
				MPI_Send(partitioned_mag.begin() + modulo_begin[node], this_size, mag_MPI_Datatype, node, 0 /* tag */, communicator);
				MPI_Send(partitioned_hash.begin() + modulo_begin[node], this_size, MPI_UNSIGNED_LONG_LONG, node, 0 /* tag */, communicator);
			}
		};

		/* 
		function to copy initial data locally
		*/
		static const auto copy_data = [&]() {
			/* resize */
			size_t this_size = modulo_begin[rank + 1] - modulo_begin[rank];
			node_map[rank].num_object = this_size;
			node_map[rank].num_object_after_interferences = 0;
			node_map[rank].resize(this_size);

			/* copy partitioned properties into node map */
			#pragma omp parallel for schedule(static)
			for (size_t i = 0; i < this_size; ++i) {
				size_t id = modulo_begin[rank] + i;

				node_map[rank].magnitude[i] = partitioned_mag[id];
				node_map[rank].hash[i] = partitioned_hash[id];
			}
		};

		/*
		function to receive final result
		*/
		static const auto receive_result = [&](int node) {
			/* receive properties */
			size_t this_size = modulo_begin[node + 1] - modulo_begin[node];
			if (this_size > 0) {
				MPI_Recv(partitioned_mag.begin() + modulo_begin[node], this_size, mag_MPI_Datatype, node, 0 /* tag */, communicator, &utils::global_status);
				MPI_Recv(partitioned_is_unique.begin() + modulo_begin[node], this_size, MPI::BOOL, node, 0 /* tag */, communicator, &utils::global_status);
			}
		};

		/*
		function to send final result
		*/
		static const auto send_result = [&](int node) {
			/* send properties */
			size_t this_size = node_map[node].num_object;
			if (this_size > 0) {
				MPI_Send(node_map[node].magnitude.begin(), this_size, mag_MPI_Datatype, node, 0 /* tag */, communicator);
				MPI_Send(node_map[node].is_unique.begin(), this_size, MPI::BOOL, node, 0 /* tag */, communicator);
			}
		};

		/*
		function to copy final result locally
		*/
		static const auto copy_result = [&]() {
			size_t this_size = modulo_begin[rank + 1] - modulo_begin[rank];
			#pragma omp for schedule(static)
			for (size_t i = 0; i < this_size; ++i) {
				size_t id = modulo_begin[rank] + i;

				partitioned_mag[id] = node_map[rank].magnitude[i];
				partitioned_is_unique[id] = node_map[rank].is_unique[i];
			}
		};

		/* !!!!!!!!!!!!!!!!
		step (4)

		Actual code :
		 !!!!!!!!!!!!!!!! */

		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		node_map.resize(size);

		/* partition nodes */
		modulo_begin = (int*)calloc(size + 1, sizeof(int));
		utils::generalized_modulo_partition(next_oid.begin(), next_oid.begin() + num_object,
			hash.begin(), modulo_begin,
			size);

		/* generate partitioned hash */
		#pragma omp parallel for schedule(static)
		for (size_t id = 0; id < num_object; ++id) {
			size_t oid = next_oid[id];

			partitioned_mag[id] = magnitude[oid];
			partitioned_hash[id] = hash[oid];
		}

		/* share back partitions using cricular permutations */
		for (int offset = 1; offset <= size / 2; ++offset) {
			/* compute neighbour nodes */
			int previous_node = (size + rank - offset) % size;
			int next_node = (size + rank + offset) % size;

			/* compute receive / send order */
			bool send_first = rank % (2*offset) < offset;

			/* send and receive data accordingly */
			if (send_first) {
				send_data(previous_node);
				receive_data(previous_node);
				if (next_node != previous_node) {
					send_data(next_node);
					receive_data(next_node);
				}
			} else {
				receive_data(next_node);
				send_data(next_node);
				if (next_node != previous_node) {
					receive_data(previous_node);
					send_data(previous_node);
				}
			}
		}

		/* copy local data */
		copy_data();

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
					#pragma omp parallel for schedule(static)
					for (size_t oid = test_size; oid < node_map[node].num_object; ++oid)
						node_map[node].is_unique[oid] = true;
			}

			/* second complete test */
			if (!fast)
				#pragma omp parallel for schedule(static)
				for (size_t oid = test_size; oid < node_map[node].num_object; ++oid) //size_t oid = oid[i];
					insert_key(oid, node);
		}

		/* share back partitions also using cricular permutations */
		for (int offset = 1; offset <= size / 2; ++offset) {
			/* compute neighbour nodes */
			int previous_node = (size + rank - offset) % size;
			int next_node = (size + rank + offset) % size;

			/* compute receive / send order */
			bool send_first = rank % (2*offset) < offset;

			/* send and receive data accordingly */
			if (send_first) {
				send_result(previous_node);
				receive_result(previous_node);
				if (next_node != previous_node) {
					send_result(next_node);
					receive_result(next_node);
				}
			} else {
				receive_result(next_node);
				send_result(next_node);
				if (next_node != previous_node) {
					receive_result(previous_node);
					send_result(previous_node);
				}
			}
		}

		/* copy local data */
		copy_result();

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

		/* clear map */
		distributed_elimination_map.clear();
	}

	/*
	distributed normalization function
	*/
	void mpi_iteration::normalize(MPI_Comm communicator) {
		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		/* !!!!!!!!!!!!!!!!
		step (8)
		 !!!!!!!!!!!!!!!! */

		node_total_proba = 0;
		total_proba = 0;

		#pragma omp parallel for reduction(+:node_total_proba)
		for (size_t oid = 0; oid < num_object; ++oid)
			node_total_proba += std::norm(magnitude[oid]);

		/* accumulate probabilities on the master node */
		if (rank == 0) {
			/* add total proba for each node */
			total_proba = node_total_proba;
			PROBA_TYPE local_total_proba;
			for (int node = 1; node < size; ++node) {
				MPI_Recv(&local_total_proba, 1, Proba_MPI_Datatype, node, 0 /* tag */, communicator, &utils::global_status);
				total_proba += local_total_proba;
			}

			/* send back total proba */
			for (int node = 1; node < size; ++node)
				MPI_Send(&total_proba, 1, Proba_MPI_Datatype, node, 0 /* tag */, communicator);
		} else {
			/* send local proba */
			MPI_Send(&node_total_proba, 1, Proba_MPI_Datatype, 0, 0 /* tag */, communicator);

			/* receive total proba */
			MPI_Recv(&total_proba, 1, Proba_MPI_Datatype, 0, 0 /* tag */, communicator, &utils::global_status);
		}
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

		/* to compute node_total_proba */
		normalize(communicator);
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
			if (rank != node_id)
				node_total_proba = 0;
	}

	/*
	equalize the number of objects across nodes
	*/
	void mpi_iteration::equalize(MPI_Comm communicator) {
		static const int alone_marker = -1;

		int size, rank;
		MPI_Comm_size(communicator, &size);
		MPI_Comm_rank(communicator, &rank);

		/* gather sizes */
		size_t *sizes;
		if (rank == 0)
			sizes = (size_t*)calloc(size, sizeof(size_t));
		MPI_Gather(&num_object, 1, MPI::LONG_LONG_INT, sizes, 1, MPI::LONG_LONG_INT, 0, communicator);

		int this_pair_id;
		if (rank == 0) {
			/* partition */
			int *pair_id = (int*)calloc(size / 2, sizeof(int));
			int alone_node = utils::make_equal_pairs(sizes, sizes + size, pair_id);

			/* tell which node is alone */
			if (alone_node >= 0) {
				MPI_Send(&alone_marker, 1, MPI::INT, alone_node, 0 /* tag */, communicator);
			}

			/* send pair idx */
			this_pair_id = pair_id[0];
			for (int i = 0; i < size / 2; ++i) {
				if (i != 0) MPI_Send(&pair_id[i], 1, MPI::INT, i, 0 /* tag */, communicator);
				MPI_Send(&i, 1, MPI::INT, pair_id[i], 0 /* tag */, communicator);
			}

		} else
			/* receive pair idx */
			MPI_Recv(&this_pair_id, 1, MPI::INT, 0, 0 /* tag */, communicator, &utils::global_status);

		/* skip if this node is alone */
		if (this_pair_id == alone_marker)
			return;

		/* get the number of objects of the respective pairs */
		size_t other_num_object;
		if (rank < size / 2) {
			MPI_Send(&num_object, 1, MPI::LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator);
			MPI_Recv(&other_num_object, 1, MPI::LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator, &utils::global_status);
		} else {
			MPI_Recv(&other_num_object, 1, MPI::LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator, &utils::global_status);
			MPI_Send(&num_object, 1, MPI::LONG_LONG_INT, this_pair_id, 0 /* tag */, communicator);
		}

		/* equalize amoung pairs */
		if (num_object == other_num_object)
			return;
		if (num_object > other_num_object) {
			size_t num_object_sent = (num_object -  other_num_object) / 2;
			send_objects(num_object_sent, this_pair_id, communicator);
		} else {
			receive_objects(this_pair_id, communicator);
		}

		/* to compute node_total_proba */
		normalize(communicator);
	}
}