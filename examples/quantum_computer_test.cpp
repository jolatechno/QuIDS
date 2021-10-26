#include "../src/iqs.hpp"
#include "../src/rules/quantum_computer.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
	iqs::rule_t *H1 = new iqs::rules::quantum_computer::hadamard(1);
	iqs::rule_t *H2 = new iqs::rules::quantum_computer::hadamard(2);
	iqs::modifier_t CNOT = iqs::rules::quantum_computer::cnot(1, 3);
	iqs::modifier_t X2 = iqs::rules::quantum_computer::Xgate(2);
	iqs::modifier_t Y0 = iqs::rules::quantum_computer::Ygate(0);
	iqs::modifier_t Z3 = iqs::rules::quantum_computer::Zgate(3);
	iqs::sy_it_t sy_it; iqs::it_t buffer;

	/* constructing a starting state with different size state */
	iqs::it_t state;
	char starting_state_1[] = {true, true, false, false};
	char starting_state_2[] = {false, true, true, false, true};
	state.append(starting_state_1, starting_state_1 + 4, 1/std::sqrt(2), 0);
	state.append(starting_state_2, starting_state_2 + 5, 0, 1/std::sqrt(2));
	std::cout << "initial_state:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, H1, buffer, sy_it);
	std::cout << "\nhadamard on second qubit:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, H2, buffer, sy_it);
	std::cout << "\nhadamard on third qubit:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, CNOT);
	std::cout << "\ncnot on fourth qubit controled by second qubit:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, X2);
	std::cout << "\nX on third qubit:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, Y0);
	std::cout << "\nY on first qubit:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, Z3);
	std::cout << "\nZ on fourth qubit:\n"; iqs::rules::quantum_computer::utils::print(state);

	iqs::simulate(state, Z3);
	iqs::simulate(state, Y0);
	iqs::simulate(state, X2);
	iqs::simulate(state, CNOT);
	iqs::simulate(state, H2, buffer, sy_it);
	iqs::simulate(state, H1, buffer, sy_it);
	std::cout << "\napplied all previous gates in reverse order:\n";  iqs::rules::quantum_computer::utils::print(state);
}