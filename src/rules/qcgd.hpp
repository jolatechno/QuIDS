#include <iostream>
#include <string>
#include <ctime>

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
		uint32_t inline &num_nodes(char* object_begin) {
			return *((uint32_t*)object_begin);
		}

		uint32_t inline &node_name_begin(char *object_begin, int node) {
			uint32_t num_nodes_ = num_nodes(object_begin);
			auto offset = 4 + 2*num_nodes_;
			return *((uint32_t*)(object_begin + offset + node*4));
		}

		uint32_t inline &node_name(char *object_begin, int node) {
			uint32_t num_nodes_ = num_nodes(object_begin);
			auto offset = 8 + 6*num_nodes_;
			return *((uint32_t*)(object_begin + offset + node*4));
		}

		bool inline &left(char *object_begin, int node) {
			return (bool&)object_begin[4 + node];
		}

		bool inline &right(char *object_begin, int node) {
			uint32_t num_nodes_ = num_nodes(object_begin);
			return (bool&)object_begin[4 + num_nodes_ + node];
		}

		void inline randomize(char *object_begin) {
			uint32_t num_nodes_ = num_nodes(object_begin);
			for (auto i = 0; i < num_nodes_; ++i) {
				left(object_begin, i) = rand() & 1;
				right(object_begin, i) = rand() & 1;
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

		bool inline node_equal(uint32_t* begin_1, uint32_t* end_1, uint32_t* begin_2, uint32_t* end_2) {
			size_t n_bit = std::distance(begin_1, end_1);
			if (n_bit != std::distance(begin_2, end_2))
				return false;

			for (auto i = 0; i < n_bit; ++i)
				if(begin_1[i] != begin_2[i])
					if (!((begin_1[i] == most_left_zero && begin_2[i] == 0) || (begin_1[i] == 0 && begin_2[i] == most_left_zero)))
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
				*(child_begin++) = *it == most_left_zero ? 0 : *it; 
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

		uint32_t inline *find_midle_and_has_most_left_zero(uint32_t *parent_begin, uint32_t *parent_end, bool &has_most_left_zero) {
			if (parent_begin[0] != open_paranthesis)
				return 0;

			has_most_left_zero = false;
			
			int parenthesis_depth = 1;
			for (auto it = parent_begin + 1; it != parent_end; ++it) {
				/* check if we're at the middle vee */
				if (parenthesis_depth == 1 && *it == vee_with_paranthesis)
					return it;

				/* check if it has most left zero */
				if (*it == most_left_zero) {
					has_most_left_zero = true;
				} else
					/* increment parenthesis depth */
					parenthesis_depth += (*it == open_paranthesis) - (*it == close_paranthesis);
			}
			return parent_end;
		}

		bool inline has_most_left_zero(uint32_t *object_begin, uint32_t *object_end) {
			for (auto *it = object_begin; it != object_end; ++it)
				if (*it = most_left_zero)
					return true;

			return false;
		}

		void inline get_operations(char *parent_begin, uint32_t node, bool &split, bool &merge) {
			merge = false;
			split = graphs::left(parent_begin, node) && graphs::right(parent_begin, node);
			if (!split && node + 1 < graphs::num_nodes(parent_begin))
				merge = graphs::left(parent_begin, node) && graphs::right(parent_begin, node + 1) && !graphs::left(parent_begin, node + 1);
		}

		uint32_t inline *merge(uint32_t *left_begin, uint32_t *left_end, uint32_t* right_begin, uint32_t *right_end, uint32_t *child_begin) {
			if (*(left_end - 1) == dot_l && *(right_end - 1) == dot_r)
				if (node_equal(left_begin, left_end - 1, right_begin, right_end - 1))
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
	}

	namespace utils {
		size_t max_print_num_graphs = -1;

		void inline make_graph(char* &object_begin, char* &object_end, uint32_t size) {
			static auto per_node_size = 2 + 2*sizeof(uint32_t);
			auto object_size = 8 + per_node_size*size;

			object_begin = new char[object_size];
			object_end = object_begin + object_size;

			graphs::num_nodes(object_begin) = size;

			graphs::node_name_begin(object_begin, 0) = 0;
			for (auto i = 0; i < size; ++i) {
				graphs::left(object_begin, i) = 0;
				graphs::right(object_begin, i) = 0;
				graphs::node_name_begin(object_begin, i + 1) = i + 1;
				graphs::node_name(object_begin, i) = i << name_offset;
			}
			graphs::node_name(object_begin, 0) = most_left_zero;
		}

		void randomize(iqs::it_t &iter) {
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.objects.begin() + iter.object_begin[gid];
				graphs::randomize(begin);
			}
		}

		void print(iqs::it_t const &iter) {
			size_t *gids = new size_t[iter.num_object];
			std::iota(gids, gids + iter.num_object, 0);
			__gnu_parallel::sort(gids, gids + iter.num_object, [&](size_t gid1, size_t gid2) {
				PROBA_TYPE r1 = iter.real[gid1];
				PROBA_TYPE i1 = iter.imag[gid1];

				PROBA_TYPE r2 = iter.real[gid2];
				PROBA_TYPE i2 = iter.imag[gid2];

				return r1*r1 + i1*i1 > r2*r2 + i2*i2;
			});

			size_t num_prints = std::min(iter.num_object, max_print_num_graphs);
			for (auto i = 0; i < num_prints; ++i) {
				size_t gid = gids[i];

				auto begin = iter.objects.begin() + iter.object_begin[gid];
				auto end = iter.objects.begin() + iter.object_begin[gid + 1];

				uint32_t num_nodes = graphs::num_nodes(begin);

				std::cout << "\t" << iter.real[gid] << (iter.imag[gid] < 0 ? " - " : " + ") << std::abs(iter.imag[gid]) << "i  ";

				for (auto i = 0; i < num_nodes; ++i) {
					auto name_begin = graphs::node_name_begin(begin, i);
					auto name_end = graphs::node_name_begin(begin, i + 1);

					std::cout << "-|" << (graphs::left(begin, i) ? "<" : " ") << "|";
					for (auto j = name_begin; j < name_end; ++j)
						switch (graphs::node_name(begin, j) & name_bitmap) {
							case scalar_node_name:
								std::cout << (graphs::node_name(begin, j) >> name_offset);
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
					std::cout << "|" << (graphs::right(begin, i) ? ">" : " ") << "|-";
				}
				std::cout << "\n";
			}
		}
	}

	void step(char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		uint32_t num_nodes = graphs::num_nodes(parent_begin);
		for (auto i = 0; i < num_nodes - 1; ++i) {
			auto i_ = num_nodes - 1 - i;
			std::swap(graphs::left(parent_begin, i), graphs::left(parent_begin, i + 1));
			std::swap(graphs::right(parent_begin, i_), graphs::right(parent_begin, i_ - 1));
		}
	}

	void reversed_step(char* parent_begin, char* parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		uint32_t num_nodes = graphs::num_nodes(parent_begin);
		for (auto i = 0; i < num_nodes - 1; ++i) {
			auto i_ = num_nodes - 1 - i;
			std::swap(graphs::right(parent_begin, i), graphs::right(parent_begin, i + 1));
			std::swap(graphs::left(parent_begin, i_), graphs::left(parent_begin, i_ - 1));
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

			uint32_t num_nodes = graphs::num_nodes(parent_begin);

			num_child = 1;
			for (int i = 0; i < num_nodes; ++i) {
				bool Xor = graphs::left(parent_begin, i) ^ graphs::right(parent_begin, i);
				if (Xor == 0)
					num_child *= 2;
			}
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			char* child_end = operations::copy(parent_begin, parent_end, child_begin);

			uint32_t num_nodes = graphs::num_nodes(parent_begin);
			for (int i = 0; i < num_nodes; ++i) {
				bool &left = graphs::left(child_begin, i);
				bool &right = graphs::right(child_begin, i);

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

			uint32_t num_nodes = graphs::num_nodes(parent_begin);

			num_child = 1;
			for (int i = 0; i < num_nodes; ++i) {
				bool Xor = graphs::left(parent_begin, i) ^ graphs::right(parent_begin, i);
				if (Xor /* == 1 */)
					num_child *= 2;
			}
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			char* child_end = operations::copy(parent_begin, parent_end, child_begin);

			uint32_t num_nodes = graphs::num_nodes(parent_begin);
			for (int i = 0; i < num_nodes; ++i) {
				bool &left = graphs::left(child_begin, i);
				bool &right = graphs::right(child_begin, i);

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

	class split_merge : public iqs::rule {
		PROBA_TYPE do_real = 1;
		PROBA_TYPE do_imag = 0;
		PROBA_TYPE do_not_real = 0;
		PROBA_TYPE do_not_imag = 0;

	public:
		split_merge(PROBA_TYPE theta, PROBA_TYPE phi = 0, PROBA_TYPE xi = 0) {
			do_real = std::sin(theta) * std::cos(phi);
			do_imag = std::sin(theta) * std::sin(phi);
			do_not_real = std::cos(theta) * std::cos(xi);
			do_not_imag = std::cos(theta) * std::sin(xi);
		}
		inline void get_num_child(char* parent_begin, char* parent_end, uint32_t &num_child, size_t &max_child_size) const override {
			max_child_size = 3*std::distance(parent_begin, parent_end);

			uint32_t num_nodes = graphs::num_nodes(parent_begin);

			bool last_merge = false;
			bool first_split = graphs::left(parent_begin, 0) && graphs::right(parent_begin, 0);
			if (!first_split && num_nodes > 1)
				last_merge = graphs::right(parent_begin, 0) && graphs::left(parent_begin, num_nodes - 1) && !graphs::right(parent_begin, num_nodes - 1);

			num_child = first_split || last_merge ? 2 : 1;

			for (int i = first_split; i < num_nodes - last_merge; ++i) {
				/* get opeartions */
				bool split, merge;
				operations::get_operations(parent_begin, i, split, merge);
				if (split || merge)
					num_child *= 2;
			}
		}
		inline char* populate_child(char* parent_begin, char* parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			uint32_t num_nodes = graphs::num_nodes(parent_begin);

			/* check for first split or last merge */
			bool last_merge = false;
			bool first_split = graphs::left(parent_begin, 0) && graphs::right(parent_begin, 0);
			if (!first_split && num_nodes > 1)
				last_merge = graphs::right(parent_begin, 0) && graphs::left(parent_begin, num_nodes - 1) && !graphs::right(parent_begin, num_nodes - 1);
			bool first_split_overflow = false;

			/* proba for first split or last merge */
			first_split &= child_id; //forget last split if it shouldn't happend
			if (first_split) {
				iqs::utils::time_equal(real, imag, do_real, do_imag);
				child_id >>= 1;
			}
			if (last_merge) {
				if (child_id & 1) {
					iqs::utils::time_equal(real, imag, do_real, -do_imag);
				} else {
					last_merge = false;
					iqs::utils::time_equal(real, imag, -do_not_real, do_not_imag);
				}
				child_id >>= 1;
			}	

			uint32_t &child_num_nodes = graphs::num_nodes(child_begin);
			child_num_nodes = num_nodes + first_split - last_merge;

			/* first path to get the final size */
			uint32_t child_id_copy = child_id;
			for (int i = first_split + last_merge; i < num_nodes - last_merge; ++i) {
				/* get opeartions */
				bool split, merge;
				operations::get_operations(parent_begin, i, split, merge);

				/* check if the operation needs to be done */
				if (merge || split) {
					if (child_id_copy & 1) {
						/* increment num nodes */
						child_num_nodes += split - merge;

						/* get proba */
						if (split) {
							iqs::utils::time_equal(real, imag, do_real, do_imag);
						} else
							iqs::utils::time_equal(real, imag, do_real, -do_imag);
					} else
						/* get proba */
						if (split) {
							iqs::utils::time_equal(real, imag, do_not_real, do_not_imag);
						} else
							iqs::utils::time_equal(real, imag, -do_not_real, do_not_imag);

					child_id_copy >>= 1;
				}
			}

			/* util variable */
			auto parent_node_name_begin = &graphs::node_name(parent_begin, 0);
			auto child_node_name_begin = &graphs::node_name(child_begin, 0);
			graphs::node_name_begin(child_begin, 0) = 0;

			/* do first split */
			uint32_t *first_split_middle;
			if (first_split) {
				bool has_most_left_zero = true;
				first_split_middle = operations::find_midle_and_has_most_left_zero(parent_node_name_begin,
					parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
					has_most_left_zero);

				if (has_most_left_zero) {
					/* set particules position */
					graphs::left(child_begin, 0) = true;
					graphs::right(child_begin, 0) = false;
					graphs::left(child_begin, 1) = false;
					graphs::right(child_begin, 1) = true;

					/* split first node */
					auto node_name_end = operations::left(parent_node_name_begin,
						parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
						child_node_name_begin,
						first_split_middle);

					graphs::node_name_begin(child_begin, 1) = std::distance(child_node_name_begin, node_name_end);

					node_name_end = operations::right(parent_node_name_begin,
						parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
						node_name_end,
						first_split_middle);

					graphs::node_name_begin(child_begin, 2) = std::distance(child_node_name_begin, node_name_end);
				} else {
					first_split_overflow = true;

					/* set particules position */
					graphs::left(child_begin, num_nodes) = true;
					graphs::right(child_begin, num_nodes) = false;
					graphs::left(child_begin, 0) = false;
					graphs::right(child_begin, 0) = true;

					/* split first node */
					auto node_name_end = operations::right(parent_node_name_begin,
						parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
						child_node_name_begin,
						first_split_middle);

					graphs::node_name_begin(child_begin, 1) = std::distance(child_node_name_begin, node_name_end);
				}
			}

			/* do last merge */
			if (last_merge) {
				graphs::left(child_begin, 0) = true;
				graphs::right(child_begin, 0) = true;

				/* merge nodes */
				auto node_name_end = operations::merge(parent_node_name_begin + graphs::node_name_begin(parent_begin, num_nodes - 1),
					parent_node_name_begin + graphs::node_name_begin(parent_begin, num_nodes),
					parent_node_name_begin,
					parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
					child_node_name_begin);

				graphs::node_name_begin(child_begin, 1) = std::distance(child_node_name_begin, node_name_end);
			}

			/* split merge every other node */
			int offset = first_split - first_split_overflow;
			for (int i = first_split + last_merge; i < num_nodes - last_merge; ++i) {

				/* get opeartions */
				bool split, merge;
				operations::get_operations(parent_begin, i, split, merge);
				
				/* check if the operation needs to be done */
				bool Do = false;
				if (merge || split) {
					Do = child_id & 1;
					if (Do) {
						if (split) {
							/* set particule position */
							graphs::left(child_begin, i + offset) = true;
							graphs::right(child_begin, i + offset) = false;
							graphs::left(child_begin, i + offset + 1) = false;
							graphs::right(child_begin, i + offset + 1) = true;

							/* split left node */
							uint32_t *middle = 0;
							auto node_name_end = operations::left(parent_node_name_begin + graphs::node_name_begin(parent_begin, i),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
								child_node_name_begin + graphs::node_name_begin(child_begin, i + offset),
								middle);

							graphs::node_name_begin(child_begin, i + 1 + offset) = std::distance(child_node_name_begin, node_name_end);

							/* split right node */
							node_name_end = operations::right(parent_node_name_begin + graphs::node_name_begin(parent_begin, i),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
								child_node_name_begin + graphs::node_name_begin(child_begin, i + offset + 1),
								middle);

							graphs::node_name_begin(child_begin, i + 1 + offset + 1) = std::distance(child_node_name_begin, node_name_end);
						} else {
							/* set particule position */
							graphs::left(child_begin, i + offset) = true;
							graphs::right(child_begin, i + offset) = true;

							/* merge nodes */
							auto node_name_end = operations::merge(parent_node_name_begin + graphs::node_name_begin(parent_begin, i),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 2),
								child_node_name_begin + graphs::node_name_begin(child_begin, i + offset));

							graphs::node_name_begin(child_begin, i + 1 + offset) = std::distance(child_node_name_begin, node_name_end);
						}

						/* increment num nodes */
						offset += split - merge;
						i += merge;
					}

					/* roll child id */
					child_id >>= 1;
				}

				if (!Do) {
					/* set particule position */
					graphs::left(child_begin, i + offset) = graphs::left(parent_begin, i);
					graphs::right(child_begin, i + offset) = graphs::right(parent_begin, i);
					
					/* copy node */
					auto node_name_end = operations::copy(parent_node_name_begin + graphs::node_name_begin(parent_begin, i),
						parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
						child_node_name_begin + graphs::node_name_begin(child_begin, i + offset));

					graphs::node_name_begin(child_begin, i + 1 + offset) = std::distance(child_node_name_begin, node_name_end);
				}
			}

			/* finish first split */
			if (first_split_overflow) {
				/* split first node */
				auto node_name_end = operations::left(parent_node_name_begin,
					parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
					child_node_name_begin + graphs::node_name_begin(child_begin, child_num_nodes - 1),
					first_split_middle);

				graphs::node_name_begin(child_begin, child_num_nodes) = std::distance(child_node_name_begin, node_name_end);
			}

			return (char*)(child_node_name_begin + graphs::node_name_begin(child_begin, child_num_nodes));
		}
	};

	namespace flags {
		typedef std::function<void(iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it)> simulator_t;

		namespace {
			std::string strip(std::string &input, std::string const separator) {
				std::string result;

				size_t end = input.find(separator);

				if (end == std::string::npos) {
					result = input;
					input = "";
				} else {
					result = input.substr(0, end);
					input.erase(0, end + separator.length());
				}

				return result;
			}

			std::string parse(std::string const input, std::string const key, std::string const separator) {
				size_t begin = input.find(key);
				if (begin == std::string::npos)
					return "";

				size_t end = input.find(separator);
				end = end == std::string::npos ? input.size() : end;

				return input.substr(begin + key.length(), end - begin);
			}

			int parse_int_with_default(std::string const input, std::string const key, std::string const separator, int Default) {
				std::string string_value = parse(input, key, separator);
				if (string_value == "")
					return Default;

				return std::atoi(string_value.c_str());
			}

			float parse_float_with_default(std::string const input, std::string const key, std::string const separator, float Default) {
				std::string string_value = parse(input, key, separator);
				if (string_value == "")
					return Default;

				return std::atof(string_value.c_str());
			}
		}

		uint read_n_iter(const char* argv) {
			std::string string_arg = argv;
			
			int n_iters = std::atoi(strip(string_arg, ",").c_str());

			std::string string_seed = parse(string_arg, "seed=", ",");
			if (string_seed != "") {
				std::srand(std::atoi(string_seed.c_str()));
			} else
				std::srand(std::time(0));
			
			return n_iters;
		}

		iqs::it_t read_state(const char* argv) {
			std::string string_args = argv;

			iqs::it_t state;

			std::string string_arg;
			while ((string_arg = strip(string_args, ";")) != "") {
				int n_node = std::atoi(strip(string_arg, ",").c_str());
				int n_graphs = parse_int_with_default(string_arg, "n_graphs=", ",", 1);
				float real = parse_float_with_default(string_arg, "real=", ",", 1);
				float imag = parse_float_with_default(string_arg, "imag=", ",", 0);

				for (auto i = 0; i < n_graphs; ++i) {
					char *begin, *end;
					utils::make_graph(begin, end, n_node);
					state.append(begin, end, real, imag);
				}
			}

			utils::randomize(state);
			state.normalize();

			return state;
		}

		simulator_t read_rule(const char* argv, hasher_t hasher=iqs::utils::default_hasher, debug_t mid_step_function=[](int){}) {
			std::string string_args = argv;

			simulator_t result = [](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it){};

			std::string string_arg;
			while ((string_arg = strip(string_args, ";")) != "") {
				std::string rule_name = strip(string_arg, ",");
				float theta = M_PI*parse_float_with_default(string_arg, "theta=", ",", 0.25);
				float phi = M_PI*parse_float_with_default(string_arg, "phi=", ",", 0);
				float xi = M_PI*parse_float_with_default(string_arg, "xi=", ",", 0);
				int n_iter = parse_int_with_default(string_arg, "n_iter=", ",", 1);

				if (rule_name == "split_merge") {
					simulator_t previous_result = result;
					split_merge rule(theta, phi, xi);
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&rule), buffer, sy_it, hasher, mid_step_function);
					};
				} else if (rule_name == "erase_create") {
					simulator_t previous_result = result;
					erase_create rule(theta, phi, xi);
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&rule), buffer, sy_it, hasher, mid_step_function);
					};
				} else if (rule_name == "erase_create") {
					simulator_t previous_result = result;
					coin rule(theta, phi, xi);
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&rule), buffer, sy_it, hasher, mid_step_function);
					};
				} else if (rule_name == "step") {
					simulator_t previous_result = result;
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, step);
					};
				} else if (rule_name == "reversed_step") {
					simulator_t previous_result = result;
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, reversed_step);
					};
				}
			}
			
			return result;
		}

		std::tuple<int, it_t, simulator_t> parse_simulation(const char* argv, hasher_t hasher=iqs::utils::default_hasher, debug_t mid_step_function=[](int){}) {
			std::string string_args = argv;

			int n_iter = read_n_iter(strip(string_args, "|").c_str());
			it_t state = read_state(strip(string_args, "|").c_str());
			simulator_t rule = read_rule(string_args.c_str(), hasher, mid_step_function);

			return {n_iter, state, rule};
		}
	}
}