#include <iostream>

#include "../quids.hpp"

namespace quids::rules::quantum_computer {
	using namespace std::complex_literals;

	namespace utils {
		void print(quids::it_t const &iter) {
			for (auto oid = 0; oid < iter.num_object; ++oid) {
				size_t size;
				mag_t mag;
				char const *begin;
				iter.get_object(oid, begin, size, mag);

				std::cout << "\t" << mag.real() << (mag.imag() < 0 ? " - " : " + ") << std::abs(mag.imag()) << "i  ";
				for (auto it = begin; it != begin + size; ++it)
					std::cout << (*it ? '1' : '0');
				std::cout << "\n";
			}
		}
	}

	modifier_t inline cnot(uint32_t control_bit, uint32_t bit) {
		return [=](char* begin, char* end, mag_t &mag) {
			begin[bit] ^= begin[control_bit];
		};
	}

	class hadamard : public quids::rule {
		size_t bit;

	public:
		hadamard(size_t bit_) : bit(bit_) {}
		inline void get_num_child(char const *parent_begin, char const *parent_end, size_t &num_child, size_t &max_child_size) const override {
			num_child = 2;
			max_child_size = std::distance(parent_begin, parent_end);
		}
		inline void populate_child(char const *parent_begin, char const *parent_end, char* const child_begin, uint32_t const child_id, size_t &size, mag_t &mag) const override {
			static const PROBA_TYPE sqrt2 = 1/std::sqrt(2);
			mag *= parent_begin[bit] && child_id ? -sqrt2 : sqrt2;

			size = std::distance(parent_begin, parent_end);
			for (auto i = 0; i < size; ++i)
				child_begin[i] = parent_begin[i];

			child_begin[bit] ^= !child_id;
		}
	};

	modifier_t inline Xgate(size_t bit) {
		return [=](char* begin, char* end, mag_t &mag) {
			begin[bit] = !begin[bit];
		};
	}

	modifier_t inline Ygate(size_t bit) {
		return [=](char* begin, char* end, mag_t &mag) {
			mag *= 1.0i;
			if (begin[bit])
				mag *= -1;

			begin[bit] = !begin[bit];
		};
	}

	modifier_t inline Zgate(size_t bit) {
		return [=](char* begin, char* end, mag_t &mag) {
			if (begin[bit])
				mag *= -1;

			begin[bit] = !begin[bit];
		};
	}
}