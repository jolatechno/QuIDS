#include "../src/iqs.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>
#include <ctime>

#define PI 3.14159265359

int main(int argc, char* argv[]) {
	iqs::tolerance = 1e-8;

	iqs::rules::qcgd::flags::read_n_iter("1");
	iqs::it_t state = iqs::rules::qcgd::flags::read_state("6");

	iqs::rule_t *erase_create = new iqs::rules::qcgd::erase_create(0.3333);
	iqs::rule_t *coin = new iqs::rules::qcgd::erase_create(0.25, 0.25);
	iqs::rule_t *split_merge = new iqs::rules::qcgd::split_merge(0.25, 0.25, 0.25);
	iqs::rule_t *reversed_split_merge = new iqs::rules::qcgd::split_merge(0.25, 0.25, -0.25);
	
	iqs::sy_it_t sy_it; iqs::it_t buffer;

	std::cout << "initial state:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, iqs::rules::qcgd::step);
	std::cout << "\nafter step:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, coin, buffer, sy_it);
	std::cout << "\nafter coin (P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, coin, buffer, sy_it);
	iqs::simulate(state, erase_create, buffer, sy_it);
	std::cout << "\nafter coin + erase_create (P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, split_merge, buffer, sy_it);
	std::cout << "\nafter split_merge(P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, iqs::rules::qcgd::step);
	iqs::simulate(state, split_merge, buffer, sy_it);
	std::cout << "\nafter step + split_merge(P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	iqs::rules::qcgd::utils::max_print_num_graphs = 10;

	iqs::simulate(state, reversed_split_merge, buffer, sy_it);
	iqs::simulate(state, iqs::rules::qcgd::reversed_step);
	iqs::simulate(state, reversed_split_merge, buffer, sy_it);
	iqs::simulate(state, erase_create, buffer, sy_it);
	iqs::simulate(state, iqs::rules::qcgd::reversed_step);
	std::cout << "\napplied all previous gates in reverse order (P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);
}