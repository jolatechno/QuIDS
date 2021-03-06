//! @cond
#include "../src/quids_mpi.hpp"
#include "../src/rules/quantum_computer.hpp"

#include <unistd.h>

void print_all(quids::mpi::mpi_it_t const &iteration, MPI_Comm comunicator) {
	int size, rank;
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	for (int i = 0; i < size; ++i) {
		usleep(1000);
		MPI_Barrier(MPI_COMM_WORLD);
		if (rank == i) {
			std::cout << "    node " << rank << "/" << size << ":\n";
			quids::rules::quantum_computer::utils::print(iteration);
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char* argv[]) {
	int master_node_id = 1;

	quids::tolerance = 1e-8;
	quids::align_byte_length = 0;

	int size, rank, provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
    if(provided < MPI_THREAD_SERIALIZED) {
        printf("The threading support level is lesser than that demanded.\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	quids::rule_t *H1 = new quids::rules::quantum_computer::hadamard(1);
	quids::rule_t *H2 = new quids::rules::quantum_computer::hadamard(2);
	quids::rule_t *H0 = new quids::rules::quantum_computer::hadamard(0);
	quids::modifier_t X2 = quids::rules::quantum_computer::Xgate(2);
	quids::mpi::mpi_sy_it_t sy_it; quids::mpi::mpi_it_t buffer;

	quids::mpi::mpi_it_t state;
	if (rank == master_node_id) {
		char starting_state_1[4] = {true, true, false, false};
		char starting_state_2[5] = {false, true, true, false, true};
		char starting_state_3[6] = {false, true, true, false, true, false};
		state.append(starting_state_1, starting_state_1 + 4, 0.5);
		state.append(starting_state_2, starting_state_2 + 5, {0, 0.5});
		state.append(starting_state_3, starting_state_3 + 6, {0.5, -0.5});
	}

	if (rank == 0) std::cout << "initial state:\n"; print_all(state, MPI_COMM_WORLD);

	quids::simulate(state, H1, buffer, sy_it);
	quids::simulate(buffer, H2, state, sy_it);
	quids::simulate(state, H0, buffer, sy_it);
	quids::simulate(buffer, X2);

	if (rank == 0) std::cout << "\napplied some gates:\n"; print_all(buffer, MPI_COMM_WORLD);

	buffer.distribute_objects(MPI_COMM_WORLD, master_node_id);

	if (rank == 0) std::cout << "\ndistributed all objects:\n"; print_all(buffer, MPI_COMM_WORLD);

	float average_size = buffer.average_value(
		(std::function<float(const char*, const char*)>)[](const char *begin, const char *end) {
			return (float)std::distance(begin, end);
		}, MPI_COMM_WORLD);
	if (rank == 0) std::cout << "\nthe average size is " << average_size << "\n";

	size_t total_num_object = buffer.get_total_num_object(MPI_COMM_WORLD);
	if (rank == 0) std::cout << "the total number of objects is " << total_num_object << "\n";

	quids::simulate(buffer, X2);
	quids::mpi::simulate(buffer, H0, state, sy_it, MPI_COMM_WORLD);
	quids::mpi::simulate(state, H2, buffer, sy_it, MPI_COMM_WORLD);
	quids::mpi::simulate(buffer, H1, state, sy_it, MPI_COMM_WORLD);

	if (rank == 0) std::cout << "\napplied all gate in reverse other (" << state.total_proba << "=P):\n"; print_all(state, MPI_COMM_WORLD);

	state.gather_objects(MPI_COMM_WORLD, master_node_id);

	if (rank == 0) std::cout << "\ngathered all objects:\n"; print_all(state, MPI_COMM_WORLD);

	MPI_Finalize();
	return 0;
}