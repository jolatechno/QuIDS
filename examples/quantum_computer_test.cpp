#include "../src/iqds.hpp"
#include "../src/rules/quantum_computer.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
	iqds::rule_t *H1 = new iqds::rules::quantum_computer::hadamard(1);
	iqds::rule_t *H2 = new iqds::rules::quantum_computer::hadamard(2);
	iqds::modifier_t CNOT = iqds::rules::quantum_computer::cnot(1, 3);
	iqds::modifier_t X2 = iqds::rules::quantum_computer::Xgate(2);
	iqds::modifier_t Y0 = iqds::rules::quantum_computer::Ygate(0);
	iqds::modifier_t Z3 = iqds::rules::quantum_computer::Zgate(3);
	iqds::sy_it_t sy_it; iqds::it_t buffer;

	/* constructing a starting state with different size state */
	iqds::it_t state;
	char starting_state_1[4] = {true, true, false, false};
	char starting_state_2[5] = {false, true, true, false, true};
	state.append(starting_state_1, starting_state_1 + 4, 1/std::sqrt(2));
	state.append(starting_state_2, starting_state_2 + 5, {0, 1/std::sqrt(2)});
	std::cout << "initial state:\n"; iqds::rules::quantum_computer::utils::print(state);

	iqds::simulate(state, H1, buffer, sy_it);
	std::cout << "\nhadamard on second qubit:\n"; iqds::rules::quantum_computer::utils::print(buffer);

	iqds::simulate(buffer, H2, state, sy_it);
	std::cout << "\nhadamard on third qubit:\n"; iqds::rules::quantum_computer::utils::print(state);

	iqds::simulate(state, CNOT);
	std::cout << "\ncnot on fourth qubit controled by second qubit:\n"; iqds::rules::quantum_computer::utils::print(state);

	iqds::simulate(state, X2);
	std::cout << "\nX on third qubit:\n"; iqds::rules::quantum_computer::utils::print(state);

	iqds::simulate(state, Y0);
	std::cout << "\nY on first qubit:\n"; iqds::rules::quantum_computer::utils::print(state);

	iqds::simulate(state, Z3);
	std::cout << "\nZ on fourth qubit:\n"; iqds::rules::quantum_computer::utils::print(state);

	iqds::simulate(state, Z3);
	iqds::simulate(state, Y0);
	iqds::simulate(state, X2);
	iqds::simulate(state, CNOT);
	iqds::simulate(state, H2, buffer, sy_it);
	iqds::simulate(buffer, H1, state, sy_it);
	std::cout << "\napplied all previous gates in reverse order:\n";  iqds::rules::quantum_computer::utils::print(state);
}