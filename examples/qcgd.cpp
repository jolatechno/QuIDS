#include "../src/lib/iqs.hpp"
#include "../src/rules/qcgd.hpp"

#include <iostream>

int main(int argc, char* argv[]) {
	char *begin, *end;
	iqs::it_t state;

	iqs::rules::qcgd::utils::make_graph(begin, end, 10);
	state.append(begin, end, 1/std::sqrt(2), 0);

	iqs::rules::qcgd::utils::make_graph(begin, end, 5);
	state.append(begin, end, 0, -1/std::sqrt(2));

	std::cout << "initial state:\n"; iqs::rules::qcgd::utils::print(state);

	iqs::rules::qcgd::utils::randomize(state);
	std::cout << "\nafter randomizing:\n"; iqs::rules::qcgd::utils::print(state);
}