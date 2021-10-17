#include <iostream>

#include "../lib/iqs.hpp"

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

	class cnot : public iqs::rule {
		size_t control_bit, bit;

	public:
		cnot(size_t cbit, size_t bit_) : control_bit(cbit), bit(bit_) {}
		inline void get_num_child(char* object_begin, char* object_end, uint16_t &num_child, size_t &max_child_size) const override {
			num_child = 1;
			max_child_size = std::distance(object_begin, object_end);
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint16_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			size_t n_bit = std::distance(parent_begin, parent_end);
			for (auto i = 0; i < n_bit; ++i)
				child_begin[i] = parent_begin[i];

			child_begin[bit] ^= child_begin[control_bit];

			return child_begin + n_bit;
		}
	};

	class hadamard : public iqs::rule {
		size_t bit;

	public:
		hadamard(size_t bit_) : bit(bit_) {}
		inline void get_num_child(char* object_begin, char* object_end, uint16_t &num_child, size_t &max_child_size) const override {
			num_child = 2;
			max_child_size = std::distance(object_begin, object_end);
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint16_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
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

	class Xgate : public iqs::rule {
		size_t n_bit, bit;

	public:
		Xgate(size_t bit_) : bit(bit_) {}
		inline void get_num_child(char* object_begin, char* object_end, uint16_t &num_child, size_t &max_child_size) const override {
			num_child = 1;
			max_child_size = std::distance(object_begin, object_end);
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint16_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			size_t n_bit = std::distance(parent_begin, parent_end);
			for (auto i = 0; i < n_bit; ++i)
				child_begin[i] = parent_begin[i];

			child_begin[bit] = !child_begin[bit];

			return child_begin + n_bit;
		}
	};

	class Ygate : public iqs::rule {
		size_t bit;

	public:
		Ygate(size_t bit_) : bit(bit_) {}
		inline void get_num_child(char* object_begin, char* object_end, uint16_t &num_child, size_t &max_child_size) const override {
			num_child = 1;
			max_child_size = std::distance(object_begin, object_end);
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint16_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			PROBA_TYPE r = -imag;
			imag = real;
			real = r;

			if (parent_begin[bit]) {
				real = -real;
				imag = -imag;
			}
			
			size_t n_bit = std::distance(parent_begin, parent_end);
			for (auto i = 0; i < n_bit; ++i)
				child_begin[i] = parent_begin[i];

			child_begin[bit] = !child_begin[bit];

			return child_begin + n_bit;
		}
	};

	class Zgate : public iqs::rule {
		size_t bit;

	public:
		Zgate(size_t bit_) : bit(bit_) {}
		inline void get_num_child(char* object_begin, char* object_end, uint16_t &num_child, size_t &max_child_size) const override {
			num_child = 1;
			max_child_size = std::distance(object_begin, object_end);
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint16_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			if (parent_begin[bit]) {
				real = -real;
				imag = -imag;
			}

			size_t n_bit = std::distance(parent_begin, parent_end);
			for (auto i = 0; i < n_bit; ++i)
				child_begin[i] = parent_begin[i];

			return child_begin + n_bit;
		}
	};
}