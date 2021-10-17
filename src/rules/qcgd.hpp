#include <iostream>

#include "../lib/iqs.hpp"

namespace iqs::rules::qcgd {
	const int name_offset = 4;

	namespace graphs {
		uint32_t inline num_nodes(char* object_begin, char* object_end) {
			return *((uint32_t*)object_begin);
		}

		void inline randomize(char *object_begin, char *object_end) {
			uint16_t num_nodes_ = num_nodes(object_begin, object_end);
			for (auto i = 0; i < num_nodes_; ++i)
				object_begin[i + 4] = rand() & 0x0003;
		}

		uint32_t inline node_name_begin(char *object_begin, char *object_end, int node) {
			if (node == 0)
				return 0;

			uint16_t num_nodes_ = num_nodes(object_begin, object_end);
			auto offset = 4 + num_nodes_;
			return *((uint32_t*)(object_begin + offset + (node - 1)*4));
		}
	}

	namespace utils {
		void inline make_graph(char* &object_begin, char* &object_end, uint32_t size) {
			static auto per_node_size = 1 + 2*sizeof(uint32_t);
			auto object_size = 4 + per_node_size*size;

			object_begin = new char[object_size];
			object_end = object_begin + object_size;

			uint32_t *object_begin_32 = (uint32_t*)object_begin;
			*object_begin_32 = size;

			uint32_t *node_name_begin = (uint32_t*)(object_begin + 4 + size);
			uint32_t *node_name = (uint32_t*)(object_begin + 4 + 5*size);
			for (auto i = 0; i < size; ++i) {
				object_begin[4 + i] = 0;
				node_name_begin[i] = i + 1;
				node_name[i] = i << name_offset;
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

				uint32_t *node_name = (uint32_t*)(begin + 4 + 5*num_nodes);
				for (auto i = 0; i < num_nodes; ++i) {
					auto name_begin = graphs::node_name_begin(begin, end, i);
					auto name_end = graphs::node_name_begin(begin, end, i + 1);

					std::cout << "-|" << (begin[4 + i] & 1 ? "<" : " ") << "|";
					for (auto j = name_begin; j < name_end; ++j)
						std::cout << (node_name[j] >> name_offset);
					std::cout << "|" << ((begin[4 + i] >> 1) & 1 ? ">" : " ") << "|-";
				}
				std::cout << "\n";
			}
		}

		void randomize(iqs::it_t &iter) {
			for (auto gid = 0; gid < iter.num_object; ++gid) {
				auto begin = iter.objects.begin() + iter.object_begin[gid];
				auto end = iter.objects.begin() + iter.object_begin[gid + 1];

				graphs::randomize(begin, end);
			}
		}
	}
}