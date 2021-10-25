#include "../src/lib/iqs.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>
#include <ctime>

#define PI 3.14159265359

int main(int argc, char* argv[]) {
	iqs::set_tolerance(1e-8);

	iqs::rules::qcgd::flags::read_n_iter("1");
	iqs::it_t state = iqs::rules::qcgd::flags::read_state("6");//;3,imag=1,real=0");

	auto erase_create = iqs::rules::qcgd::flags::read_rule("erase_create,theta=0.3333");
	auto coin = iqs::rules::qcgd::flags::read_rule("erase_create,theta=0.25,phi=0.25");
	auto split_merge = iqs::rules::qcgd::flags::read_rule("split_merge,theta=0.25,phi=0.25");
	auto step = iqs::rules::qcgd::flags::read_rule("step");
	auto reversed_step = iqs::rules::qcgd::flags::read_rule("reversed_step");

	iqs::sy_it_t sy_it; iqs::it_t buffer;

	std::cout << "initial state:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::simulate(state, iqs::rules::qcgd::step);
	std::cout << "\nafter step:\n"; iqs::rules::qcgd::utils::print(state);

	coin(state, buffer, sy_it);
	std::cout << "\nafter coin (P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	coin(state, buffer, sy_it);
	erase_create(state, buffer, sy_it);
	std::cout << "\nafter coin + erase_create (P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	split_merge(state, buffer, sy_it);
	std::cout << "\nafter split_merge(P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	step(state, buffer, sy_it);
	split_merge(state, buffer, sy_it);
	std::cout << "\nafter step + split_merge(P=" << state.total_proba << "):\n"; iqs::rules::qcgd::utils::print(state);

	split_merge(state, buffer, sy_it);
	reversed_step(state, buffer, sy_it);
	split_merge(state, buffer, sy_it);
	erase_create(state, buffer, sy_it);
	reversed_step(state, buffer, sy_it);
	std::cout << "\napplied all previous gates in reverse order (P=" << state.total_proba << "):\n";  iqs::rules::qcgd::utils::print(state);
}