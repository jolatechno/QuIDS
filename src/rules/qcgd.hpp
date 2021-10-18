#include <iostream>

#include "../lib/iqs.hpp"

namespace iqs::rules::qcgd {
	const int name_offset = 4;

	namespace graphs {
		uint32_t inline &num_nodes(char* object_begin, char* object_end) {
			return *((uint32_t*)object_begin);
		}

		uint32_t inline &node_name_begin(char *object_begin, char *object_end, int node) {
			uint32_t num_nodes_ = num_nodes(object_begin, object_end);
			auto offset = 4 + 2*num_nodes_;
			return *((uint32_t*)(object_begin + offset + node*4));
		}

		uint32_t inline &node_name(char *object_begin, char *object_end, int node) {
			uint32_t num_nodes_ = num_nodes(object_begin, object_end);
			auto offset = 8 + 6*num_nodes_;
			return *((uint32_t*)(object_begin + offset + node*4));
		}

		bool inline &left(char *object_begin, char *object_end, int node) {
			return (bool&)object_begin[4 + node];
		}

		bool inline &right(char *object_begin, char *object_end, int node) {
			uint32_t num_nodes_ = num_nodes(object_begin, object_end);
			return (bool&)object_begin[4 + num_nodes_ + node];
		}

		void inline randomize(char *object_begin, char *object_end) {
			uint32_t num_nodes_ = num_nodes(object_begin, object_end);
			for (auto i = 0; i < num_nodes_; ++i) {
				left(object_begin, object_end, i) = rand() & 1;
				right(object_begin, object_end, i) = rand() & 1;
			}
		}
	}

	namespace opeartions {
		char inline *merge(char *left_begin, char *left_end, char* right_begin, char *right_end, char *child_begin) {
			return child_begin;
		}

		char inline *left(char *parent_begin, char *parent_end, char *child_begin) {
			return child_begin;
		}

		char inline *right(char *parent_begin, char *parent_end, char *child_begin) {
			return child_begin;
		}
	}

	namespace utils {
		void inline make_graph(char* &object_begin, char* &object_end, uint32_t size) {
			static auto per_node_size = 2 + 2*sizeof(uint32_t);
			auto object_size = 8 + per_node_size*size;

			object_begin = new char[object_size];
			object_end = object_begin + object_size;

			graphs::num_nodes(object_begin, object_end) = size;

			graphs::node_name_begin(object_begin, object_end, 0) = 0;
			for (auto i = 0; i < size; ++i) {
				graphs::left(object_begin, object_end, i) = 0;
				graphs::right(object_begin, object_end, i) = 0;
				graphs::node_name_begin(object_begin, object_end, i + 1) = i + 1;
				graphs::node_name(object_begin, object_end, i) = i << name_offset;
			}
		}

		void randomize(iqs::it_t &iter) {
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.objects.begin() + iter.object_begin[gid];
				auto end = iter.objects.begin() + iter.object_begin[gid + 1];

				graphs::randomize(begin, end);
			}
		}

		/*
		needs rework !!!
		*/
		void print(iqs::it_t const &iter) {
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.objects.begin() + iter.object_begin[gid];
				auto end = iter.objects.begin() + iter.object_begin[gid + 1];

				uint32_t num_nodes = graphs::num_nodes(begin, end);

				std::cout << "\t" << iter.real[gid] << (iter.imag[gid] < 0 ? " - " : " + ") << std::abs(iter.imag[gid]) << "i  ";

				for (auto i = 0; i < num_nodes; ++i) {
					auto name_begin = graphs::node_name_begin(begin, end, i);
					auto name_end = graphs::node_name_begin(begin, end, i + 1);

					std::cout << "-|" << (graphs::left(begin, end, i) ? "<" : " ") << "|";
					for (auto j = name_begin; j < name_end; ++j)
						std::cout << (graphs::node_name(begin, end, j) >> name_offset);
					std::cout << "|" << (graphs::right(begin, end, i) ? ">" : " ") << "|-";
				}
				std::cout << "\n";
			}
		}
	}

	void step(char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		uint32_t num_nodes = graphs::num_nodes(parent_begin, parent_end);
		for (auto i = 0; i < num_nodes - 1; ++i) {
			auto i_ = num_nodes - 1 - i;
			std::swap(graphs::left(parent_begin, parent_end, i), graphs::left(parent_begin, parent_end, i + 1));
			std::swap(graphs::right(parent_begin, parent_end, i_), graphs::right(parent_begin, parent_end, i_ - 1));
		}
	}

	void reversed_step(char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		uint32_t num_nodes = graphs::num_nodes(parent_begin, parent_end);
		for (auto i = 0; i < num_nodes - 1; ++i) {
			auto i_ = num_nodes - 1 - i;
			std::swap(graphs::right(parent_begin, parent_end, i), graphs::right(parent_begin, parent_end, i + 1));
			std::swap(graphs::left(parent_begin, parent_end, i_), graphs::left(parent_begin, parent_end, i_ - 1));
		}
	}

	class erase_create : public iqs::rule {
		PROBA_TYPE do_real = 1;
		PROBA_TYPE do_imag = 0;
		PROBA_TYPE do_not_real = 0;
		PROBA_TYPE do_not_imag = 0;

	public:
		erase_create(PROBA_TYPE theta, PROBA_TYPE phi = 0, PROBA_TYPE xi = 0) {
			do_real = std::sin(theta) * std::cos(phi);
			do_imag = std::sin(theta) * std::sin(phi);
			do_not_real = std::cos(theta) * std::cos(xi);
			do_not_imag = std::cos(theta) * std::sin(xi);
		}
		inline void get_num_child(char* parent_begin, char* parent_end, uint32_t &num_child, size_t &max_child_size) const override {
			max_child_size = std::distance(parent_begin, parent_end);

			uint32_t num_nodes = graphs::num_nodes(parent_begin, parent_end);

			num_child = 1;
			for (int i = 0; i < num_nodes; ++i) {
				bool Xor = graphs::left(parent_begin, parent_end, i) ^ graphs::right(parent_begin, parent_end, i);
				if (Xor == 0)
					num_child *= 2;
			}
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			size_t n_bit = std::distance(parent_begin, parent_end);
			char* child_end = child_begin + n_bit;
			for (auto i = 0; i < n_bit; ++i)
				child_begin[i] = parent_begin[i];

			uint32_t num_nodes = graphs::num_nodes(parent_begin, parent_end);
			for (int i = 0; i < num_nodes; ++i) {
				bool &left = graphs::left(child_begin, child_end, i);
				bool &right = graphs::right(child_begin, child_end, i);

				uint8_t sum = left + right;
				if ((sum & 1) == 0) {
					PROBA_TYPE sign = 1 - sum;
					if (child_id & 1) {
						iqs::utils::time_equal(real, imag, do_real, sign*do_imag);
						left = !left;
						right = !right;
					} else
						iqs::utils::time_equal(real, imag, sign*do_not_real, do_not_imag);
					child_id >>= 1;
				}
			}

			return child_end;
		}
	};
}