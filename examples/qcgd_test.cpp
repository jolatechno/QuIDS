#include "../src/iqds.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>
#include <ctime>

#define PI 3.14159265359

int main(int argc, char* argv[]) {
	iqds::tolerance = 1e-15;

	iqds::sy_it_t sy_it;
	iqds::it_t state, buffer;

	iqds::rules::qcgd::flags::read_n_iter("1");
	iqds::rules::qcgd::flags::read_state("6", state);

	iqds::rule_t *erase_create = new iqds::rules::qcgd::erase_create(0.3333);
	iqds::rule_t *coin = new iqds::rules::qcgd::erase_create(0.25, 0.25);
	iqds::rule_t *split_merge = new iqds::rules::qcgd::split_merge(0.25, 0.25, 0.25);
	iqds::rule_t *reversed_split_merge = new iqds::rules::qcgd::split_merge(0.25, 0.25, -0.25);

	std::cout << "initial state:\n"; iqds::rules::qcgd::utils::print(state);

	iqds::simulate(state, iqds::rules::qcgd::step);
	std::cout << "\nafter step:\n"; iqds::rules::qcgd::utils::print(state);

	iqds::simulate(state, coin, buffer, sy_it);
	std::cout << "\nafter coin (P=" << buffer.total_proba << "):\n"; iqds::rules::qcgd::utils::print(buffer);

	iqds::simulate(buffer, coin, state, sy_it);
	iqds::simulate(state, erase_create, buffer, sy_it);
	std::cout << "\nafter coin + erase_create (P=" << buffer.total_proba << "):\n"; iqds::rules::qcgd::utils::print(buffer);

	iqds::simulate(buffer, split_merge, state, sy_it);
	std::cout << "\nafter split_merge(P=" << state.total_proba << "):\n"; iqds::rules::qcgd::utils::print(state);

	iqds::simulate(state, iqds::rules::qcgd::step);
	iqds::simulate(state, split_merge, buffer, sy_it);
	std::cout << "\nafter step + split_merge(P=" << buffer.total_proba << "):\n"; iqds::rules::qcgd::utils::print(buffer);

	iqds::rules::qcgd::utils::max_print_num_graphs = 10;

	iqds::simulate(buffer, reversed_split_merge, state, sy_it);
	iqds::simulate(state, iqds::rules::qcgd::reversed_step);
	iqds::simulate(state, reversed_split_merge, buffer, sy_it);
	iqds::simulate(buffer, erase_create, state, sy_it);
	iqds::simulate(state, iqds::rules::qcgd::reversed_step);
	std::cout << "\napplied all previous gates in reverse order (P=" << state.total_proba << "):\n"; iqds::rules::qcgd::utils::print(state);
}