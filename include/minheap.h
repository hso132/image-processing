#ifndef MIN_HEAP_H
#define MIN_HEAP_H

#include <vector>
#include <ostream>
#include <limits>

namespace sapph_dijkstra
{
	//the node index type; for better clarity
	typedef size_t node_index_t;


	//the internal Node class
	class Node
	{
		public:
			/**
			 * @brief parameter constructor
			 * @param neighbours the neighbouring nodes' indices
			 * @param node_idx the node's index
			 * @param dist the distance (for the dijkstra algorithm; defaults to infinity)
			 */
			Node(const std::vector<node_index_t> &neighbours, node_index_t node_idx, double dist=std::numeric_limits<double>::infinity());
			friend std::ostream& operator<<(std::ostream& stream, const Node& n);
			double distance() const;
			std::vector<node_index_t> neighbours() const;
			node_index_t index() const;
			void set_distance(double dist);
			//Node() = default;
		protected:
			friend class MinHeap;
			std::vector<node_index_t> m_neighbours;
			double m_distance;
			node_index_t node_index;
	};

	template <class T>
		class Array
		{
			public:
				Array(size_t size);
				Array(const std::vector<T> &data);

				T& operator[](size_t idx);
				const T& operator[](size_t idx) const;
			protected:
				std::vector<T> m_data;

		};
	class MinHeap
	{
		typedef size_t heap_index_t;
		public:
			MinHeap(const std::vector<Node> &nodes);
			bool decrease_key(node_index_t node_idx, double new_distance);
			Node* extract_min();
			inline bool is_empty(){return heap_size==0;};
			friend std::ostream& operator<<(std::ostream& stream, const MinHeap& heap);
			Array<Node> nodes() const;
		protected:

			size_t heap_size;
			size_t num_nodes;


			Array<heap_index_t> internal_indices; //a mapping of node index to heap index
			Array<Node> m_nodes; //the actual nodes
			Array<node_index_t> heap; //the heap; stores only the node indices, but orders by node distance

			/**
			 * Max heapifies the heap at the given root; assumes that both 
			 * the left and right subtrees are min heaps already
			 */
			void min_heapify(heap_index_t root); 

			/***************************************************
			inline helper functions
			***************************************************/

			//gets the node at a certain point in the heap
			inline Node* get_node(heap_index_t heap_idx)
			{
				if(heap_idx < heap_size)
				{
					node_index_t node_idx = heap[heap_idx];
					return &(m_nodes[0])+node_idx;
				}
				else
				{
					return nullptr;
				}
			};
			//gets the left offspring heap index
			inline heap_index_t left(heap_index_t p) { return 2*p; };

			//gets the right offspring heap index
			inline heap_index_t right(heap_index_t p) {return left(p)+1;};

			inline heap_index_t parent(heap_index_t c) {return c/2;};

			inline void swap_heap_elems(heap_index_t idx1, heap_index_t idx2)
			{
				node_index_t node1 = heap[idx1];
				node_index_t node2 = heap[idx2]; 
				heap[idx1] = node2;
				heap[idx2] = node1;
				internal_indices[node1] = idx2;
				internal_indices[node2] = idx1;
			}

	};
}

#endif
