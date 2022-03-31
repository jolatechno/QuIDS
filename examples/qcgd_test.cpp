#include "../src/quids.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>
#include <ctime>

#define PI 3.14159265359

int main(int argc, char* argv[]) {
	quids::tolerance = 1e-15;

	quids::sy_it_t sy_it;
	quids::it_t state, buffer;

	quids::rules::qcgd::flags::read_n_iter("1");
	quids::rules::qcgd::flags::read_state("6", state);

	quids::rule_t *erase_create = new quids::rules::qcgd::erase_create(0.3333);
	quids::rule_t *coin = new quids::rules::qcgd::erase_create(0.25, 0.25);
	quids::rule_t *split_merge = new quids::rules::qcgd::split_merge(0.25, 0.25, 0.25);
	quids::rule_t *reversed_split_merge = new quids::rules::qcgd::split_merge(0.25, 0.25, -0.25);

	std::cout << "initial state:\n"; quids::rules::qcgd::utils::print(state);

	quids::simulate(state, quids::rules::qcgd::step);
	std::cout << "\nafter step:\n"; quids::rules::qcgd::utils::print(state);

	quids::simulate(state, coin, buffer, sy_it);
	std::cout << "\nafter coin (P=" << buffer.total_proba << "):\n"; quids::rules::qcgd::utils::print(buffer);

	quids::simulate(buffer, coin, state, sy_it);
	quids::simulate(state, erase_create, buffer, sy_it);
	std::cout << "\nafter coin + erase_create (P=" << buffer.total_proba << "):\n"; quids::rules::qcgd::utils::print(buffer);

	quids::simulate(buffer, split_merge, state, sy_it);
	std::cout << "\nafter split_merge(P=" << state.total_proba << "):\n"; quids::rules::qcgd::utils::print(state);

	quids::simulate(state, quids::rules::qcgd::step);
	quids::simulate(state, split_merge, buffer, sy_it);
	std::cout << "\nafter step + split_merge(P=" << buffer.total_proba << "):\n"; quids::rules::qcgd::utils::print(buffer);

	quids::rules::qcgd::utils::max_print_num_graphs = 10;

	quids::simulate(buffer, reversed_split_merge, state, sy_it);
	quids::simulate(state, quids::rules::qcgd::reversed_step);
	quids::simulate(state, reversed_split_merge, buffer, sy_it);
	quids::simulate(buffer, erase_create, state, sy_it);
	quids::simulate(state, quids::rules::qcgd::reversed_step);
	std::cout << "\napplied all previous gates in reverse order (P=" << state.total_proba << "):\n"; quids::rules::qcgd::utils::print(state);
}