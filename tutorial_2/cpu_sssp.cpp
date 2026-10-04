#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace std;

const int INF = numeric_limits<int>::max() / 4;

struct CSRGraph {
    vector<int> rowOffsets;
    vector<int> columnIndices;
    vector<int> weights;
};

CSRGraph readGraph(const string& fileName, int& vertexCount) {
    ifstream input(fileName);
    if (!input) {
        throw runtime_error("Could not open graph file: " + fileName);
    }

    int edgeCount;
    input >> vertexCount >> edgeCount;
    if (!input || vertexCount <= 0 || edgeCount < 0) {
        throw runtime_error("First line must be: <number of vertices> <number of edges>");
    }

    vector<int> from(edgeCount);
    vector<int> to(edgeCount);
    vector<int> weight(edgeCount);
    vector<int> degree(vertexCount, 0);

    for (int edge = 0; edge < edgeCount; ++edge) {
        input >> from[edge] >> to[edge] >> weight[edge];
        if (!input || from[edge] < 0 || from[edge] >= vertexCount ||
            to[edge] < 0 || to[edge] >= vertexCount || weight[edge] < 0) {
            throw runtime_error("Invalid edge. Vertices must be valid and weights non-negative.");
        }
        degree[from[edge]]++;
    }

    CSRGraph graph;
    graph.rowOffsets.resize(vertexCount + 1, 0);
    for (int vertex = 0; vertex < vertexCount; ++vertex) {
        graph.rowOffsets[vertex + 1] = graph.rowOffsets[vertex] + degree[vertex];
    }

    graph.columnIndices.resize(edgeCount);
    graph.weights.resize(edgeCount);
    vector<int> nextPosition = graph.rowOffsets;
    for (int edge = 0; edge < edgeCount; ++edge) {
        int position = nextPosition[from[edge]]++;
        graph.columnIndices[position] = to[edge];
        graph.weights[position] = weight[edge];
    }
    return graph;
}

vector<int> bellmanFord(const CSRGraph& graph, int source) {
    int vertexCount = static_cast<int>(graph.rowOffsets.size()) - 1;
    vector<int> distance(vertexCount, INF);
    distance[source] = 0;

    for (int iteration = 0; iteration < vertexCount - 1; ++iteration) {
        bool changed = false;
        vector<int> nextDistance = distance;

        for (int vertex = 0; vertex < vertexCount; ++vertex) {
            if (distance[vertex] == INF) {
                continue;
            }
            for (int edge = graph.rowOffsets[vertex];
                 edge < graph.rowOffsets[vertex + 1]; ++edge) {
                int neighbor = graph.columnIndices[edge];
                int candidate = distance[vertex] + graph.weights[edge];
                if (candidate < nextDistance[neighbor]) {
                    nextDistance[neighbor] = candidate;
                    changed = true;
                }
            }
        }

        distance = nextDistance;
        if (!changed) {
            break;
        }
    }
    return distance;
}

int main(int argc, char** argv) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <graph-file>\n";
        return 1;
    }

    try {
        int vertexCount;
        CSRGraph graph = readGraph(argv[1], vertexCount);
        vector<int> distance = bellmanFord(graph, 0);

        cout << "CPU Bellman-Ford distances from vertex 0:\n";
        for (int vertex = 0; vertex < vertexCount; ++vertex) {
            cout << vertex << ": ";
            if (distance[vertex] == INF) {
                cout << "INF\n";
            } else {
                cout << distance[vertex] << '\n';
            }
        }
    } catch (const exception& error) {
        cerr << "Error: " << error.what() << '\n';
        return 1;
    }
    return 0;
}