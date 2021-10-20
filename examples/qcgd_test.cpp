#include "../src/lib/iqs.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>
#include <ctime>

#define PI 3.14159265359

int main(int argc, char* argv[]) {
	auto seed = std::time(0);
	std::srand(seed);
	std::cout << seed << "=seed\n";

	iqs::rule_t* erase_create = new iqs::rules::qcgd::erase_create(PI / 3);
	iqs::rule_t* coin = new iqs::rules::qcgd::coin(PI / 4, PI / 4);
	iqs::rule_t* split_merge = new iqs::rules::qcgd::split_merge(PI / 4, 0);
	iqs::sy_it_t sy_it; iqs::it_t buffer;

	char *begin, *end;
	iqs::it_t state;

	iqs::rules::qcgd::utils::make_graph(begin, end, 4);
	state.append(begin, end, 1/std::sqrt(2), 0);

	iqs::rules::qcgd::utils::make_graph(begin, end, 3);
	state.append(begin, end, 0, -1/std::sqrt(2));

	std::cout << "initial state:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::rules::qcgd::utils::randomize(state);
	std::cout << "\nafter randomizing:\n"; iqs::rules::qcgd::utils::print(state);

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

	iqs::simulate(state, split_merge, buffer, sy_it);
	iqs::simulate(state, iqs::rules::qcgd::reversed_step);
	iqs::simulate(state, split_merge, buffer, sy_it);
	iqs::simulate(state, erase_create, buffer, sy_it);
	iqs::simulate(state, iqs::rules::qcgd::reversed_step);
	std::cout << "\napplied all previous gates in reverse order (P=" << state.total_proba << "):\n";  iqs::rules::qcgd::utils::print(state);
}