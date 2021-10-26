#include <iostream>

#include "../iqs.hpp"

namespace iqs::rules::quantum_computer {
	namespace utils {
		void print(iqs::it_t const &iter) {
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.objects.begin() + iter.object_begin[gid];
				auto end = iter.objects.begin() + iter.object_begin[gid + 1];

				std::cout << "\t" << iter.real[gid] << (iter.imag[gid] < 0 ? " - " : " + ") << std::abs(iter.imag[gid]) << "i  ";
				for (auto it = begin; it != end; ++it)
					std::cout << (*it ? '1' : '0');
				std::cout << "\n";
			}
		}
	}

	modifier_t inline cnot(uint32_t control_bit, uint32_t bit) {
		return [=](char* begin, char* end, PROBA_TYPE &real, PROBA_TYPE &imag) {
			begin[bit] ^= begin[control_bit];
		};
	}

	class hadamard : public iqs::rule {
		size_t bit;

	public:
		hadamard(size_t bit_) : bit(bit_) {}
		inline void get_num_child(char* parent_begin, char* parent_end, uint32_t &num_child, size_t &max_child_size) const override {
			num_child = 2;
			max_child_size = std::distance(parent_begin, parent_end);
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			static const PROBA_TYPE sqrt2 = 1/std::sqrt(2);
			PROBA_TYPE multiplier = parent_begin[bit] && child_id ? -sqrt2 : sqrt2;
			real *= multiplier; imag *= multiplier;

			size_t n_bit = std::distance(parent_begin, parent_end);
			for (auto i = 0; i < n_bit; ++i)
				child_begin[i] = parent_begin[i];

			child_begin[bit] ^= !child_id;

			return child_begin + n_bit;
		}
	};

	modifier_t inline Xgate(size_t bit) {
		return [=](char* begin, char* end, PROBA_TYPE &real, PROBA_TYPE &imag) {
			begin[bit] = !begin[bit];
		};
	}

	modifier_t inline Ygate(size_t bit) {
		return [=](char* begin, char* end, PROBA_TYPE &real, PROBA_TYPE &imag) {
			PROBA_TYPE r = -imag;
			imag = real;
			real = r;

			if (begin[bit]) {
				real = -real;
				imag = -imag;
			}

			begin[bit] = !begin[bit];
		};
	}

	modifier_t inline Zgate(size_t bit) {
		return [=](char* begin, char* end, PROBA_TYPE &real, PROBA_TYPE &imag) {
			if (begin[bit]) {
				real = -real;
				imag = -imag;
			}

			begin[bit] = !begin[bit];
		};
	}
}