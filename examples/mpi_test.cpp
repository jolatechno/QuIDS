#include "../src/iqs_mpi.hpp"
#include "../src/rules/quantum_computer.hpp"

void print_all(iqs::mpi::mpi_it_t const &iteration, MPI_Comm comunicator) {
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	char synchronizer = 0;

	if (rank == 0) {
		std::cout << "  node 0/" << size << ":\n";
		iqs::rules::quantum_computer::utils::print(iteration);

		MPI_Send(&synchronizer, 1, MPI_CHAR, 1, 0 /* tag */, MPI_COMM_WORLD);
		MPI_Recv(&synchronizer, 1, MPI_CHAR, size - 1, 0 /* tag */, MPI_COMM_WORLD, &iqs::mpi::utils::global_status);
	} else {
		int next_node = (rank + 1) % size;
		int last_node = rank - 1;
		MPI_Recv(&synchronizer, 1, MPI_CHAR, last_node, 0 /* tag */, MPI_COMM_WORLD, &iqs::mpi::utils::global_status);

		std::cout << "  node " << rank << "/" << size << ":\n";
		iqs::rules::quantum_computer::utils::print(iteration);

		MPI_Send(&synchronizer, 1, MPI_CHAR, next_node, 0 /* tag */, MPI_COMM_WORLD);
	}
}

int main(int argc, char* argv[]) {
	iqs::set_tolerance(1e-8);

	int size, rank;
    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	iqs::rule_t *H1 = new iqs::rules::quantum_computer::hadamard(1);
	iqs::rule_t *H2 = new iqs::rules::quantum_computer::hadamard(2);
	iqs::rule_t *H3 = new iqs::rules::quantum_computer::hadamard(3);
	iqs::modifier_t X2 = iqs::rules::quantum_computer::Xgate(2);
	iqs::mpi::mpi_sy_it_t sy_it; iqs::mpi::mpi_it_t buffer;

	iqs::mpi::mpi_it_t state;
	if (rank == 0) {
		char starting_state_1[4] = {true, true, false, false};
		char starting_state_2[5] = {false, true, true, false, true};
		state.append(starting_state_1, starting_state_1 + 4, 1/std::sqrt(2), 0);
		state.append(starting_state_2, starting_state_2 + 5, 0, 1/std::sqrt(2));
	}
	if (rank == 0) std::cout << "initial state:\n"; print_all(state, MPI_COMM_WORLD);

	iqs::simulate(state, H1, buffer, sy_it);
	iqs::simulate(state, H2, buffer, sy_it);
	iqs::simulate(state, H3, buffer, sy_it);
	iqs::simulate(state, X2);

	if (rank == 0) std::cout << "\napplied some gates:\n"; print_all(state, MPI_COMM_WORLD);
	
	state.distribute_objects(MPI_COMM_WORLD);

	if (rank == 0) std::cout << "\ndistributed all objects:\n"; print_all(state, MPI_COMM_WORLD);

	iqs::simulate(state, X2);
	iqs::mpi::simulate(state, H3, buffer, sy_it, MPI_COMM_WORLD);
	iqs::mpi::simulate(state, H2, buffer, sy_it, MPI_COMM_WORLD);
	iqs::mpi::simulate(state, H1, buffer, sy_it, MPI_COMM_WORLD);

	if (rank == 0) std::cout << "\napplied all gate in reverse other:\n"; print_all(state, MPI_COMM_WORLD);

	state.gather_objects(MPI_COMM_WORLD);

	if (rank == 0) std::cout << "\ngathered all objects:\n"; print_all(state, MPI_COMM_WORLD);

	MPI_Finalize();
}