#include <iostream>

#include "../lib/iqs.hpp"

namespace iqs::rules::qcgd {
	const int name_offset = 3;
	const uint32_t name_bitmap = (1 << name_offset) - 1;
	enum {
		scalar_node_name = 0,
		most_left_zero,
		vee_with_paranthesis,
		open_paranthesis,
		close_paranthesis,
		dot_l,
		dot_r
	};

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

	namespace operations {
		template<class T>
		bool inline equal(T* begin_1, T* end_1, T* begin_2, T* end_2) {
			size_t n_bit = std::distance(begin_1, end_1);
			if (n_bit != std::distance(begin_2, end_2))
				return false;

			for (auto i = 0; i < n_bit; ++i)
				if(begin_1[i] != begin_2[i])
					return false;

			return true;
		}

		template<class T>
		T inline *copy(T *parent_begin, T* parent_end, T* child_begin) {
			for (auto it = parent_begin; it != parent_end; ++it)
				*(child_begin++) = *it;
			return child_begin;
		}

		uint32_t inline *copy_without_most_left_zero(uint32_t *parent_begin, uint32_t *parent_end, uint32_t *child_begin) {
			for (auto it = parent_begin; it != parent_end; ++it)
				*(child_begin++) = *it == most_left_zero ? 1 << name_offset : *it; 
			return child_begin;
		}

		uint32_t inline *find_midle(uint32_t *parent_begin, uint32_t *parent_end) {
			if (parent_begin[0] != open_paranthesis)
				return 0;
			
			int parenthesis_depth = 1;
			for (auto it = parent_begin + 1; it != parent_end; ++it) {
				if (parenthesis_depth == 1 && *it == vee_with_paranthesis)
					return it;
				parenthesis_depth += (*it == open_paranthesis) - (*it == close_paranthesis);
			}
			return parent_end;
		}
			
		uint32_t inline *merge(uint32_t *left_begin, uint32_t *left_end, uint32_t* right_begin, uint32_t *right_end, uint32_t *child_begin) {
			if (*(left_end - 1) == dot_l && *(right_end - 1) == dot_r)
				if (equal(left_begin, left_end - 1, right_begin, right_end - 1))
					return copy(left_begin, left_end - 1, child_begin);
			
			*(child_begin++) = open_paranthesis;
			child_begin = copy(left_begin, left_end, child_begin);
			*(child_begin++) = vee_with_paranthesis;
			child_begin = copy(right_begin, right_end, child_begin);
			*(child_begin++) = close_paranthesis;
			
			return child_begin;
		}

		uint32_t inline *left(uint32_t *parent_begin, uint32_t *parent_end, uint32_t *child_begin, uint32_t *&middle) {
			if (parent_begin[0] == open_paranthesis) {
				if (middle == 0)
					middle = find_midle(parent_begin, parent_end);

				child_begin = copy(parent_begin + 1, middle, child_begin);
			} else {
				child_begin = copy(parent_begin, parent_end, child_begin);
				*(child_begin++) = dot_l;
			}
			return child_begin;
		}
		uint32_t inline *left(uint32_t *parent_begin, uint32_t *parent_end, uint32_t *child_begin) {
			uint32_t *middle = 0;
			return left(parent_begin, parent_end, child_begin, middle);
		}

		uint32_t inline *right(uint32_t *parent_begin, uint32_t *parent_end, uint32_t *child_begin, uint32_t *&middle) {
			if (parent_begin[0] == open_paranthesis) {
				if (middle == 0)
					middle = find_midle(parent_begin, parent_end);

				child_begin = copy(middle + 1, parent_end - 1, child_begin);
			} else {
				child_begin = copy_without_most_left_zero(parent_begin, parent_end, child_begin);
				*(child_begin++) = dot_r;
			}
			return child_begin;
		}
		uint32_t inline *right(uint32_t *parent_begin, uint32_t *parent_end, uint32_t *child_begin) {
			uint32_t *middle = 0;
			return right(parent_begin, parent_end, child_begin, middle);
		}

		bool has_most_left_zero(uint32_t *object_begin, uint32_t *object_end) {
			for (auto *it = object_begin; it != object_end; ++it)
				if (*it = most_left_zero)
					return true;

			return false;
		}
	}

	namespace utils {
		void inline make_graph(char* &object_begin, char* &object_end, uint32_t size) {
			static auto per_node_size = 2 + 2*sizeof(uint32_t);
			auto object_size = 8 + per_node_size*size /* for testing */ + 10*size;

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
			graphs::node_name(object_begin, object_end, 0) = most_left_zero;

			/* 
			testing:
			*/
			uint32_t *node_name_begin = &graphs::node_name(object_begin, object_end, 0);
			/*
			testing merge
			*/
			uint32_t left[] = {(uint32_t)(0 << name_offset)};
			uint32_t right[] = {(uint32_t)(1 << name_offset)};
			uint32_t* begin = &graphs::node_name(object_begin, object_end, size - 5);
			uint32_t* end = operations::merge(left, left + 1, right, right + 1, begin);
			graphs::node_name_begin(object_begin, object_end, size - 4) = std::distance(node_name_begin, end);
			/*
			testing splits (left, simple)
			*/
			uint32_t node[] = {(uint32_t)(2 << name_offset)};
			begin = end;
			end = operations::left(node, node + 1, begin);
			graphs::node_name_begin(object_begin, object_end, size - 3) = std::distance(node_name_begin, end);
			/*
			testing splits (right, simple)
			*/
			begin = end;
			end = operations::right(node, node + 1, begin);
			graphs::node_name_begin(object_begin, object_end, size - 2) = std::distance(node_name_begin, end);
			/*
			testing splits (complicated)
			*/
			uint32_t *node_begin = node_name_begin + graphs::node_name_begin(object_begin, object_end, size - 5);
			uint32_t *node_end = node_name_begin + graphs::node_name_begin(object_begin, object_end, size - 4);
			/*
			testing splits (left, complicated)
			*/
			begin = end;
			end = operations::left(node_begin, node_end, begin);
			graphs::node_name_begin(object_begin, object_end, size - 1) = std::distance(node_name_begin, end);
			/*
			testing splits (right, complicated)
			*/
			begin = end;
			end = operations::right(node_begin, node_end, begin);
			graphs::node_name_begin(object_begin, object_end, size) = std::distance(node_name_begin, end);


			object_end = (char*)end + 4;
		}

		void randomize(iqs::it_t &iter) {
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.objects.begin() + iter.object_begin[gid];
				auto end = iter.objects.begin() + iter.object_begin[gid + 1];

				graphs::randomize(begin, end);
			}
		}

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
						switch (graphs::node_name(begin, end, j) & name_bitmap) {
							case scalar_node_name:
								std::cout << (graphs::node_name(begin, end, j) >> name_offset);
								break;
							case most_left_zero:
								std::cout << "0*";
								break;
							case vee_with_paranthesis:
								std::cout << ")∨(";
								break;
							case open_paranthesis:
								std::cout << "(";
								break;
							case close_paranthesis:
								std::cout << ")";
								break;
							case dot_l:
								std::cout << ".l";
								break;
							case dot_r:
								std::cout << ".r";
								break;
						}
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
			char* child_end = operations::copy(parent_begin, parent_end, child_begin);

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

	class coin : public iqs::rule {
		PROBA_TYPE do_real = 1;
		PROBA_TYPE do_imag = 0;
		PROBA_TYPE do_not_real = 0;
		PROBA_TYPE do_not_imag = 0;

	public:
		coin(PROBA_TYPE theta, PROBA_TYPE phi = 0, PROBA_TYPE xi = 0) {
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
				if (Xor /* == 1 */)
					num_child *= 2;
			}
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			char* child_end = operations::copy(parent_begin, parent_end, child_begin);

			uint32_t num_nodes = graphs::num_nodes(parent_begin, parent_end);
			for (int i = 0; i < num_nodes; ++i) {
				bool &left = graphs::left(child_begin, child_end, i);
				bool &right = graphs::right(child_begin, child_end, i);

				if (left ^ right /* == 1 */) {
					PROBA_TYPE sign = 1 - 2*left;
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