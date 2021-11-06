#include "../src/iqs_mpi.hpp"
#include "../src/rules/quantum_computer.hpp"

int main(int argc, char* argv[]) {
	iqs::set_tolerance(1e-8);

	iqs::rule_t *H1 = new iqs::rules::quantum_computer::hadamard(1);
	iqs::modifier_t X2 = iqs::rules::quantum_computer::Xgate(2);
	iqs::mpi::mpi_sy_it_t sy_it; iqs::mpi::mpi_it_t buffer;

	iqs::mpi::mpi_it_t state;
	char starting_state_1[4] = {true, true, false, false};
	char starting_state_2[5] = {false, true, true, false, true};
	state.append(starting_state_1, starting_state_1 + 4, 1/std::sqrt(2), 0);
	state.append(starting_state_2, starting_state_2 + 5, 0, 1/std::sqrt(2));
	std::cout << "initial state:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, H1, buffer, sy_it);
	iqs::simulate(state, X2);

	iqs::simulate(state, X2);
	iqs::mpi::simulate(state, H1, buffer, sy_it, MPI_COMM_WORLD);

	std::cout << "\nfinal state:\n"; iqs::rules::quantum_computer::utils::print(state);
}