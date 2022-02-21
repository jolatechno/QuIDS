#include "../src/iqs.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>
#include <ctime>

#define PI 3.14159265359

int main(int argc, char* argv[]) {
	iqs::tolerance = 1e-15;

	iqs::sy_it_t sy_it;
	iqs::it_t state, buffer;

	iqs::rules::qcgd::flags::read_n_iter("1");
	iqs::rules::qcgd::flags::read_state("6", state);

	iqs::rule_t *erase_create = new iqs::rules::qcgd::erase_create(0.3333);
	iqs::rule_t *coin = new iqs::rules::qcgd::erase_create(0.25, 0.25);
	iqs::rule_t *split_merge = new iqs::rules::qcgd::split_merge(0.25, 0.25, 0.25);
	iqs::rule_t *reversed_split_merge = new iqs::rules::qcgd::split_merge(0.25, 0.25, -0.25);

	std::cout << "initial state:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, iqs::rules::qcgd::step);
	std::cout << "\nafter step:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, coin, buffer, sy_it);
	std::cout << "\nafter coin (P=" << buffer.total_proba << "):\n"; iqs::rules::qcgd::utils::print(buffer);

	iqs::simulate(buffer, coin, state, sy_it);
	iqs::simulate(state, erase_create, buffer, sy_it);
	std::cout << "\nafter coin + erase_create (P=" << buffer.total_proba << "):\n"; iqs::rules::qcgd::utils::print(buffer);

	iqs::simulate(buffer, split_merge, state, sy_it);
	std::cout << "\nafter split_merge(P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, iqs::rules::qcgd::step);
	iqs::simulate(state, split_merge, buffer, sy_it);
	std::cout << "\nafter step + split_merge(P=" << buffer.total_proba << "):\n"; iqs::rules::qcgd::utils::print(buffer);

	iqs::rules::qcgd::utils::max_print_num_graphs = 10;

	iqs::simulate(buffer, reversed_split_merge, state, sy_it);
	iqs::simulate(state, iqs::rules::qcgd::reversed_step);
	iqs::simulate(state, reversed_split_merge, buffer, sy_it);
	iqs::simulate(buffer, erase_create, state, sy_it);
	iqs::simulate(state, iqs::rules::qcgd::reversed_step);
	std::cout << "\napplied all previous gates in reverse order (P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);
}