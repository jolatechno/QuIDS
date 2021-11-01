#include <iostream>
#include <string>
#include <ctime>

#include "../iqs.hpp"

namespace iqs::rules::qcgd {
	namespace utils {
		inline void hash_combine(std::size_t& seed, size_t const value_64) {
		    static size_t magic_number = 0xc6a4a7935bd1e995;
		    static auto shift = 47;

		    seed *= magic_number;
		    seed ^= value_64 >> shift;
		    seed *= magic_number;

		    seed ^= value_64;
		    seed *= magic_number;

		    // Completely arbitrary number, to prevent 0's
		    // from hashing to 0.
		    seed += 0xe6546b64;
	    }
	}

	namespace graphs {
		enum {
			dot_l_t = -3,
			dot_r_t,
			element_t,
			pair_t
		};
		struct sub_node {
			int16_t hmlz_and_element;
			int16_t right_or_type;
			size_t hash;

			sub_node(int16_t element) : right_or_type(element_t), hash(element) {
				hmlz_and_element = element == 0 ? -1 : element + 1;
			}
			sub_node(sub_node const &node, int16_t type) : right_or_type(type) {
				if (type == dot_l_t && node.hmlz_and_element < 0) {
					hmlz_and_element = -1;
				} else
					hmlz_and_element = 1;

				hash = node.hash;
				utils::hash_combine(hash, type);
			}
			sub_node(sub_node const &left, sub_node const &right, int16_t right_offset) : right_or_type(right_offset) {
				if (left.hmlz_and_element < 0 || right.hmlz_and_element < 0) {
					hmlz_and_element = -1;
				} else
					hmlz_and_element = 1;

				hash = left.hash;
				utils::hash_combine(hash, right.hash);
			}
		};

		uint16_t inline &num_nodes(char *object_begin) {
			return *((uint16_t*)object_begin);
		}

		uint16_t inline const num_nodes(char const *object_begin) {
			return *((uint16_t*)object_begin);
		}

		uint16_t inline *node_name_begin(char *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			auto offset = sizeof(uint16_t) + 2*num_nodes_;
			return (uint16_t*)(object_begin + offset);
		}
		uint16_t inline const *node_name_begin(char const *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			auto offset = sizeof(uint16_t) + 2*num_nodes_;
			return (const uint16_t*)(object_begin + offset);
		}
		uint16_t inline &node_name_begin(char *object_begin, int node) { return node_name_begin(object_begin)[node]; }
		uint16_t inline const node_name_begin(char const *object_begin, int node) { return node_name_begin(object_begin)[node]; }


		sub_node inline *node_name(char *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			auto offset = 2*sizeof(uint16_t) + (2 + sizeof(uint16_t))*num_nodes_;
			return (sub_node*)(object_begin + offset);
		}
		sub_node inline const *node_name(char const *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			auto offset = 2*sizeof(uint16_t) + (2 + sizeof(uint16_t))*num_nodes_;
			return (const sub_node*)(object_begin + offset);
		} 
		sub_node inline &node_name(char *object_begin, int node) { return node_name(object_begin)[node]; }
		sub_node inline const node_name(char const *object_begin, int node) { return node_name(object_begin)[node]; }

		bool inline *left(char *object_begin) { return (bool*)(object_begin + sizeof(uint16_t)); }
		bool inline const *left(char const *object_begin) { return (const bool*)(object_begin + sizeof(uint16_t)); }
		bool inline &left(char *object_begin, int node) { return left(object_begin)[node]; }
		bool inline const left(char const *object_begin, int node) { return left(object_begin)[node]; }

		bool inline *right(char *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			return (bool*)(object_begin + sizeof(uint16_t) + num_nodes_);
		}
		bool inline const *right(char const *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			return (const bool*)(object_begin + sizeof(uint16_t) + num_nodes_);
		}
		bool inline &right(char *object_begin, int node) { return right(object_begin)[node]; }
		bool inline const right(char const *object_begin, int node) { return right(object_begin)[node]; }

		void inline randomize(char *object_begin) {
			uint16_t num_nodes_ = num_nodes(object_begin);
			for (auto i = 0; i < num_nodes_; ++i) {
				left(object_begin, i) = rand() & 1;
				right(object_begin, i) = rand() & 1;
			}
		}

		size_t inline hash_graph(char const *object_begin) {
			size_t left_hash = 0;
			size_t right_hash = 0;
			size_t name_hash = 0;

			bool const *left_ = left(object_begin);
			bool const *right_ = right(object_begin);
			auto const *node_begin = node_name_begin(object_begin);
			auto const *node_name_ = node_name(object_begin);

			uint16_t const num_nodes_ = num_nodes(object_begin);
			for (auto i = 0; i < num_nodes_; ++i) {
				if (left_[i])
					utils::hash_combine(left_hash, i);

				if (right_[i])
					utils::hash_combine(right_hash, i);

				utils::hash_combine(name_hash, node_name_[node_begin[i]].hash);
			}

			utils::hash_combine(name_hash, left_hash);
			utils::hash_combine(name_hash, right_hash);
			return name_hash;
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
		T inline *copy(T const *parent_begin, T const *parent_end, T* child_begin) {
			for (auto it = parent_begin; it != parent_end; ++it)
				*(child_begin++) = *it;
			return child_begin;
		}

		bool inline has_most_left_zero(graphs::sub_node *object_begin) {
			return object_begin->hmlz_and_element < 0;
		}

		void inline get_operations(char const *parent_begin, uint16_t node, bool &split, bool &merge) {
			merge = false;
			split = graphs::left(parent_begin, node) && graphs::right(parent_begin, node);
			if (!split && node + 1 < graphs::num_nodes(parent_begin))
				merge = graphs::left(parent_begin, node) && graphs::right(parent_begin, node + 1) && !graphs::left(parent_begin, node + 1);
		}

		graphs::sub_node inline *merge(graphs::sub_node const *left_begin, graphs::sub_node const *left_end,
			graphs::sub_node const *right_begin, graphs::sub_node const *right_end,
			graphs::sub_node *child_begin) {

			if (left_begin->right_or_type == graphs::dot_l_t && right_begin->right_or_type == graphs::dot_r_t)
				if ((left_begin + 1)->hash == (right_begin + 1)->hash)
					return copy(left_begin + 1, left_end, child_begin);
			
			*(child_begin++) = graphs::sub_node(*left_begin, *right_begin, std::distance(left_begin, left_end) + 1);
			child_begin = copy(left_begin, left_end, child_begin);
			return copy(right_begin, right_end, child_begin);
		}

		graphs::sub_node inline *left(graphs::sub_node const *parent_begin, graphs::sub_node const *parent_end, graphs::sub_node *child_begin) {
			if (parent_begin->right_or_type >= 0) 
				return copy(parent_begin + 1, parent_begin + parent_begin->right_or_type, child_begin);

			*(child_begin++) = graphs::sub_node(*parent_begin, graphs::dot_l_t);
			return copy(parent_begin, parent_end, child_begin);
		}

		graphs::sub_node inline *right(graphs::sub_node const *parent_begin, graphs::sub_node const *parent_end, graphs::sub_node *child_begin) {
			if (parent_begin->right_or_type >= 0)
				return copy(parent_begin + parent_begin->right_or_type, parent_end, child_begin);

			*(child_begin++) = graphs::sub_node(*parent_begin, graphs::dot_r_t);
			return copy(parent_begin, parent_end, child_begin);
		}
	}

	namespace utils {
		namespace {
			size_t max_print_num_graphs = -1;
		}

		void set_max_print_num_graphs(size_t val) { max_print_num_graphs = val; }

		void make_graph(char* &object_begin, char* &object_end, uint16_t size) {
			static auto per_node_size = 2 + sizeof(uint16_t) + sizeof(graphs::sub_node);
			auto object_size = 2*sizeof(uint16_t) + per_node_size*size;

			object_begin = new char[object_size];
			object_end = object_begin + object_size;

			graphs::num_nodes(object_begin) = size;

			graphs::node_name_begin(object_begin, 0) = 0;
			for (auto i = 0; i < size; ++i) {
				graphs::left(object_begin, i) = 0;
				graphs::right(object_begin, i) = 0;
				graphs::node_name_begin(object_begin, i + 1) = i + 1;
				graphs::node_name(object_begin, i) = graphs::sub_node(i);
			}
		}

		void randomize(iqs::it_t &iter) {
			size_t size;
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.get_object(gid, size);
				graphs::randomize(begin);
			}
		}

		void print(iqs::it_t const &iter) {
			static std::function<void(graphs::sub_node const*)> print_node_name = [](graphs::sub_node const *sub_node) {
				if (sub_node->right_or_type == graphs::element_t) {
					std::cout << std::abs(sub_node->hmlz_and_element) - 1;

				} else if (sub_node->right_or_type == graphs::dot_l_t) {
					std::cout << "(";
					print_node_name(sub_node + 1);
					std::cout << ").l";

				} else if (sub_node->right_or_type == graphs::dot_r_t) {
					std::cout << "(";
					print_node_name(sub_node + 1);
					std::cout << ").r";

				} else {
					std::cout << "(";
					print_node_name(sub_node + 1);
					std::cout << ")∨(";
					print_node_name(sub_node + sub_node->right_or_type);
					std::cout << ")";
				}
			};

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

				size_t size;
				const char* begin = iter.get_object(gid, size);

				uint16_t const num_nodes = graphs::num_nodes(begin);

				PROBA_TYPE real = std::abs(iter.real[gid]) < iqs::tolerance ? 0 : iter.real[gid];
				PROBA_TYPE imag = std::abs(iter.imag[gid]) < iqs::tolerance ? 0 : iter.imag[gid];

				std::cout << "\t" << real << (imag < 0 ? " - " : " + ") << std::abs(imag) << "i  ";

				for (auto i = 0; i < num_nodes; ++i) {
					auto name_begin = graphs::node_name_begin(begin, i);

					std::cout << "-|" << (graphs::left(begin, i) ? "<" : " ") << "|";
					print_node_name(graphs::node_name(begin) + name_begin);
					std::cout << "|" << (graphs::right(begin, i) ? ">" : " ") << "|-";
				}
				std::cout << "\n";
			}

			if (num_prints < iter.num_object)
				std::cout << "\t...and " << iter.num_object - num_prints << " other graphs\n";
		}

		void serialize(iqs::it_t const &iter, iqs::sy_it_t const &sy_it, uint indentation=0) {
			double interference_ratio = 1;
			double deletion_ratio = 1; 
			if (sy_it.num_object > 0) {
				interference_ratio = ((double)sy_it.num_object_after_interferences) / ((double)sy_it.num_object);
				deletion_ratio = ((double)iter.num_object) / ((double)sy_it.num_object_after_interferences);
			}

			double avg_size = 0;
			double avg_squared_size = 0;
			double avg_density = 0;
			double avg_squared_density = 0;

			for (auto gid = 0; gid < iter.num_object; ++gid) {
				size_t size;
				const char* begin = iter.get_object(gid, size);

				uint16_t const num_nodes = graphs::num_nodes(begin);
				double double_num_nodes = num_nodes;

				PROBA_TYPE r = iter.real[gid];
				PROBA_TYPE i = iter.imag[gid];
				double proba = r*r + i*i;

				double density = 0;
				for (auto i = 0; i < num_nodes; ++i)
					density += graphs::left(begin, i); + graphs::right(begin, i);
				density /= 2*double_num_nodes;

				avg_size += proba*double_num_nodes;
				avg_squared_size += proba*double_num_nodes*double_num_nodes;
				avg_density += proba*density;
				avg_squared_density += proba*density*density;
			}

			double std_dev_size = std::sqrt(avg_squared_size - avg_size*avg_size);
			double std_dev_density = std::sqrt(avg_squared_density - avg_density*avg_density);

			auto const print_indentation = [=]() {
				for (auto i = 0; i < indentation; ++i)
					std::cout << "\t";
			};

			std::cout << "{\n";
			print_indentation(); std::cout << "\t\"total_proba\" : " << iter.total_proba << ",\n";
			print_indentation(); std::cout << "\t\"num_graphs\" : " << iter.num_object << ",\n";
			print_indentation(); std::cout << "\t\"avg_size\" : " << avg_size << ",\n";
			print_indentation(); std::cout << "\t\"std_dev_size\" : " << std_dev_size << ",\n";
			print_indentation(); std::cout << "\t\"avg_density\" : " << avg_density << ",\n";
			print_indentation(); std::cout << "\t\"std_dev_density\" : " << std_dev_density << ",\n";
			print_indentation(); std::cout << "\t\"interference_ratio\" : " << interference_ratio << ",\n";
			print_indentation(); std::cout << "\t\"deletion_ratio\" : " << deletion_ratio << "\n";
			print_indentation(); std::cout << "}";
		}
	}

	void step(char *parent_begin, char *parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		uint16_t num_nodes = graphs::num_nodes(parent_begin);
		auto left_ = graphs::left(parent_begin);
		auto right_ = graphs::right(parent_begin);
		std::rotate(left_, left_ + 1, left_ + num_nodes);
		std::rotate(right_, right_ + num_nodes - 1, right_ + num_nodes);
	}

	void reversed_step(char *parent_begin, char *parent_end, PROBA_TYPE &real, PROBA_TYPE &imag) {
		uint16_t num_nodes = graphs::num_nodes(parent_begin);
		auto left_ = graphs::left(parent_begin);
		auto right_ = graphs::right(parent_begin);
		std::rotate(right_, right_ + 1, right_ + num_nodes);
		std::rotate(left_, left_ + num_nodes - 1, left_ + num_nodes);
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
		inline size_t hasher(char const *parent_begin, char const *parent_end) const override {
			return graphs::hash_graph(parent_begin);
		}
		inline void get_num_child(char const *parent_begin, char const *parent_end, uint32_t &num_child, size_t &max_child_size) const override {
			max_child_size = std::distance(parent_begin, parent_end);

			uint16_t num_nodes = graphs::num_nodes(parent_begin);

			num_child = 1;
			for (int i = 0; i < num_nodes; ++i) {
				bool Xor = graphs::left(parent_begin, i) ^ graphs::right(parent_begin, i);
				if (Xor == 0)
					num_child *= 2;
			}
		}
		inline char* populate_child(char const *parent_begin, char const *parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			char* child_end = operations::copy(parent_begin, parent_end, child_begin);

			uint16_t num_nodes = graphs::num_nodes(parent_begin);
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
		inline size_t hasher(char const *parent_begin, char const *parent_end) const override {
			return graphs::hash_graph(parent_begin);
		}
		inline void get_num_child(char const *parent_begin, char const *parent_end, uint32_t &num_child, size_t &max_child_size) const override {
			max_child_size = std::distance(parent_begin, parent_end);

			uint16_t num_nodes = graphs::num_nodes(parent_begin);

			num_child = 1;
			for (int i = 0; i < num_nodes; ++i) {
				bool Xor = graphs::left(parent_begin, i) ^ graphs::right(parent_begin, i);
				if (Xor /* == 1 */)
					num_child *= 2;
			}
		}
		inline char* populate_child(char const *parent_begin, char const *parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			char* child_end = operations::copy(parent_begin, parent_end, child_begin);

			uint16_t num_nodes = graphs::num_nodes(parent_begin);
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
		inline size_t hasher(char const *parent_begin, char const *parent_end) const override {
			return graphs::hash_graph(parent_begin);
		}
		inline void get_num_child(char const *parent_begin, char const *parent_end, uint32_t &num_child, size_t &max_child_size) const override {
			max_child_size = 4*std::distance(parent_begin, parent_end);

			uint16_t num_nodes = graphs::num_nodes(parent_begin);

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
		inline char* populate_child(char const *parent_begin, char const *parent_end, uint32_t child_id, PROBA_TYPE &real, PROBA_TYPE &imag, char* child_begin) const override {
			uint16_t num_nodes = graphs::num_nodes(parent_begin);

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

			uint16_t &child_num_nodes = graphs::num_nodes(child_begin);
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
			auto parent_node_name_begin = graphs::node_name(parent_begin);
			auto child_node_name_begin = graphs::node_name(child_begin);
			graphs::node_name_begin(child_begin, 0) = 0;

			/* do first split */
			if (first_split) {
				bool has_most_left_zero = true;
				if (parent_node_name_begin->right_or_type >= 0)
					if ((parent_node_name_begin + 1)->hmlz_and_element > 0)
						has_most_left_zero = false;

				if (has_most_left_zero) {
					/* set particules position */
					graphs::left(child_begin, 0) = true;
					graphs::right(child_begin, 0) = false;
					graphs::left(child_begin, 1) = false;
					graphs::right(child_begin, 1) = true;

					/* split first node */
					auto node_name_end = operations::left(parent_node_name_begin,
						parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
						child_node_name_begin);

					graphs::node_name_begin(child_begin, 1) = std::distance(child_node_name_begin, node_name_end);

					node_name_end = operations::right(parent_node_name_begin,
						parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
						node_name_end);

					graphs::node_name_begin(child_begin, 2) = std::distance(child_node_name_begin, node_name_end);
				} else {
					first_split_overflow = true;

					/* set particules position */
					graphs::left(child_begin, child_num_nodes - 1) = true;
					graphs::right(child_begin, child_num_nodes - 1) = false;
					graphs::left(child_begin, 0) = false;
					graphs::right(child_begin, 0) = true;

					/* split first node */
					auto node_name_end = operations::right(parent_node_name_begin,
						parent_node_name_begin + graphs::node_name_begin(parent_begin, 1),
						child_node_name_begin);

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
							auto node_name_end = operations::left(parent_node_name_begin + graphs::node_name_begin(parent_begin, i),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
								child_node_name_begin + graphs::node_name_begin(child_begin, i + offset));

							graphs::node_name_begin(child_begin, i + 1 + offset) = std::distance(child_node_name_begin, node_name_end);

							/* split right node */
							node_name_end = operations::right(parent_node_name_begin + graphs::node_name_begin(parent_begin, i),
								parent_node_name_begin + graphs::node_name_begin(parent_begin, i + 1),
								child_node_name_begin + graphs::node_name_begin(child_begin, i + offset + 1));

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
					child_node_name_begin + graphs::node_name_begin(child_begin, child_num_nodes - 1));

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

		std::tuple<uint, uint> read_n_iter(const char* argv) {
			std::string string_arg = argv;
			
			int n_iters = std::atoi(strip(string_arg, ",").c_str());

			std::string string_seed = parse(string_arg, "seed=", ",");
			if (string_seed != "") {
				std::srand(std::atoi(string_seed.c_str()));
			} else
				std::srand(std::time(0));

			int reversed_n_iters = parse_int_with_default(string_arg, "reversed_n_iter=", ",", n_iters);

			int max_print_num_graphs = parse_int_with_default(string_arg, "max_print_num_graphs=", ",", -1);
			utils::set_max_print_num_graphs(max_print_num_graphs);

			float tolerance = parse_float_with_default(string_arg, "tolerance=", ",", TOLERANCE);
			iqs::set_tolerance(tolerance);

			float safety_margin = parse_float_with_default(string_arg, "safety_margin=", ",", SAFETY_MARGIN);
			iqs::set_safety_margin(safety_margin);

			float collision_test_proportion = parse_float_with_default(string_arg, "collision_test_proportion=", ",", COLLISION_TEST_PROPORTION);
			iqs::set_collision_test_proportion(collision_test_proportion);

			float collision_tolerance = parse_float_with_default(string_arg, "collision_tolerance=", ",", COLLISION_TOLERANCE);
			iqs::set_collision_tolerance(collision_tolerance);
			
			return {n_iters, reversed_n_iters};
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

		std::tuple<simulator_t, simulator_t> read_rule(const char* argv, debug_t mid_step_function=[](int){}) {
			std::string string_args = argv;

			simulator_t result = [](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it){};
			simulator_t reversed_result = [](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it){};

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
							iqs::simulate(it, (rule_t*)(&rule), buffer, sy_it, mid_step_function);
					};

					split_merge reversed_rule(theta, phi, -xi);
					simulator_t previous_reversed_result = reversed_result;
					reversed_result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&reversed_rule), buffer, sy_it, mid_step_function);
						previous_reversed_result(it, buffer, sy_it);
					};

				} else if (rule_name == "erase_create") {
					erase_create rule(theta, phi, xi);
					simulator_t previous_result = result;
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&rule), buffer, sy_it, mid_step_function);
					};

					erase_create reversed_rule(theta, phi, -xi);
					simulator_t previous_reversed_result = reversed_result;
					reversed_result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&reversed_rule), buffer, sy_it, mid_step_function);
						previous_reversed_result(it, buffer, sy_it);
					};

				} else if (rule_name == "coin") {
					coin rule(theta, phi, xi);
					simulator_t previous_result = result;
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&rule), buffer, sy_it, mid_step_function);
					};

					coin reversed_rule(theta, phi, -xi);
					simulator_t previous_reversed_result = reversed_result;
					reversed_result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, (rule_t*)(&reversed_rule), buffer, sy_it, mid_step_function);
						previous_reversed_result(it, buffer, sy_it);
					};

				} else if (rule_name == "step") {
					simulator_t previous_result = result;
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, step);
					};

					simulator_t previous_reversed_result = reversed_result;
					reversed_result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, reversed_step);
						previous_reversed_result(it, buffer, sy_it);
					};

				} else if (rule_name == "reversed_step") {
					simulator_t previous_result = result;
					result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						previous_result(it, buffer, sy_it);
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, reversed_step);
					};

					simulator_t previous_reversed_result = reversed_result;
					reversed_result = [=](iqs::it_t &it, iqs::it_t &buffer, iqs::sy_it_t &sy_it) {
						for (auto i = 0; i < n_iter; ++i)
							iqs::simulate(it, step);
						previous_reversed_result(it, buffer, sy_it);
					};
				}
			}
			
			return {result, reversed_result};
		}

		std::tuple<uint, uint, it_t, simulator_t, simulator_t> parse_simulation(const char* argv, debug_t mid_step_function=[](int){}) {
			std::string string_args = argv;

			auto [n_iter, reversed_n_iters] = read_n_iter(strip(string_args, "|").c_str());
			it_t state = read_state(strip(string_args, "|").c_str());
			auto [rule, reversed_rule] = read_rule(string_args.c_str(), mid_step_function);

			return {n_iter, reversed_n_iters, state, rule, reversed_rule};
		}
	}
}