//! @cond
#include "../src/quids.hpp"
#include "../src/rules/quantum_computer.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
	quids::rule_t *H1 = new quids::rules::quantum_computer::hadamard(1);
	quids::rule_t *H2 = new quids::rules::quantum_computer::hadamard(2);
	quids::modifier_t CNOT = quids::rules::quantum_computer::cnot(1, 3);
	quids::modifier_t X2 = quids::rules::quantum_computer::Xgate(2);
	quids::modifier_t Y0 = quids::rules::quantum_computer::Ygate(0);
	quids::modifier_t Z3 = quids::rules::quantum_computer::Zgate(3);
	quids::sy_it_t sy_it; quids::it_t buffer;

	/* constructing a starting state with different size state */
	quids::it_t state;
	char starting_state_1[4] = {true, true, false, false};
	char starting_state_2[5] = {false, true, true, false, true};
	state.append(starting_state_1, starting_state_1 + 4, 1/std::sqrt(2));
	state.append(starting_state_2, starting_state_2 + 5, {0, 1/std::sqrt(2)});
	std::cout << "initial state:\n"; quids::rules::quantum_computer::utils::print(state);

	quids::simulate(state, H1, buffer, sy_it);
	std::cout << "\nhadamard on second qubit:\n"; quids::rules::quantum_computer::utils::print(buffer);

	quids::simulate(buffer, H2, state, sy_it);
	std::cout << "\nhadamard on third qubit:\n"; quids::rules::quantum_computer::utils::print(state);

	quids::simulate(state, CNOT);
	std::cout << "\ncnot on fourth qubit controled by second qubit:\n"; quids::rules::quantum_computer::utils::print(state);

	quids::simulate(state, X2);
	std::cout << "\nX on third qubit:\n"; quids::rules::quantum_computer::utils::print(state);

	quids::simulate(state, Y0);
	std::cout << "\nY on first qubit:\n"; quids::rules::quantum_computer::utils::print(state);

	quids::simulate(state, Z3);
	std::cout << "\nZ on fourth qubit:\n"; quids::rules::quantum_computer::utils::print(state);

	quids::simulate(state, Z3);
	quids::simulate(state, Y0);
	quids::simulate(state, X2);
	quids::simulate(state, CNOT);
	quids::simulate(state, H2, buffer, sy_it);
	quids::simulate(buffer, H1, state, sy_it);
	std::cout << "\napplied all previous gates in reverse order:\n";  quids::rules::quantum_computer::utils::print(state);
}