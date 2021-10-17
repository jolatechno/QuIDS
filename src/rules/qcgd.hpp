#include "../lib/iqs.hpp"

namespace iqs::rules::qcgd {
	namespace graphs {
		uint32_t inline num_nodes(char* object_begin, char* object_end) {
			return *((uint32_t*)object_begin);
		}

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
				node_name[i] = i << 4;
			}
		}

		void inline randomize(char *object_begin, char *object_end) {
			uint16_t num_nodes_ = num_nodes(object_begin, object_end);
			for (auto i = 0; i < num_nodes_; ++i)
				object_begin[i + 6] = rand() && 0x0004;
		}

		uint32_t inline node_name_begin(char *object_begin, char *object_end, int node) {
			if (node == 0)
				return 0;

			uint16_t num_nodes_ = num_nodes(object_begin, object_end);
			auto offset = 4 + num_nodes_;
			return *((uint32_t*)(object_begin + offset + (node - 1)*4));
		}
	}
}