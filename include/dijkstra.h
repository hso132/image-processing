#ifndef DIJKSTRA_H_INCLUDED
#define DIJKSTRA_H_INCLUDED
#include <vector>
#include "minheap.h"

namespace sapph_dijkstra
{
	std::vector<node_index_t> compute_path(const std::vector<Node> &graph, const std::vector<double> &weight, node_index_t source, node_index_t destination);
}

#endif
