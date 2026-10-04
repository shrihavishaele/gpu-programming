#include <cuda_runtime.h>

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

void checkCuda(cudaError_t error, const char* message) {
    if (error != cudaSuccess) {
        throw runtime_error(string(message) + ": " + cudaGetErrorString(error));
    }
}

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

__global__ void relaxEdges(const int* rowOffsets, const int* columnIndices,
                           const int* weights, int vertexCount,
                           const int* oldDistance, int* newDistance,
                           int* changed) {
    int vertex = blockIdx.x * blockDim.x + threadIdx.x;
    if (vertex >= vertexCount) {
        return;
    }

    if (oldDistance[vertex] == INF) {
        return;
    }

    for (int edge = rowOffsets[vertex]; edge < rowOffsets[vertex + 1]; ++edge) {
        int neighbor = columnIndices[edge];
        int candidate = oldDistance[vertex] + weights[edge];
        int previous = atomicMin(&newDistance[neighbor], candidate);
        if (candidate < previous) {
            *changed = 1;
        }
    }
}

vector<int> bellmanFord(const CSRGraph& graph, int source) {
    int vertexCount = static_cast<int>(graph.rowOffsets.size()) - 1;
    int edgeCount = static_cast<int>(graph.columnIndices.size());
    int* deviceRows = nullptr;
    int* deviceColumns = nullptr;
    int* deviceWeights = nullptr;
    int* deviceOldDistance = nullptr;
    int* deviceNewDistance = nullptr;
    int* deviceChanged = nullptr;

    try {
        checkCuda(cudaMalloc(&deviceRows, (vertexCount + 1) * sizeof(int)), "Allocating rows");
        checkCuda(cudaMalloc(&deviceColumns, edgeCount * sizeof(int)), "Allocating columns");
        checkCuda(cudaMalloc(&deviceWeights, edgeCount * sizeof(int)), "Allocating weights");
        checkCuda(cudaMalloc(&deviceOldDistance, vertexCount * sizeof(int)), "Allocating old distances");
        checkCuda(cudaMalloc(&deviceNewDistance, vertexCount * sizeof(int)), "Allocating new distances");
        checkCuda(cudaMalloc(&deviceChanged, sizeof(int)), "Allocating changed flag");

        checkCuda(cudaMemcpy(deviceRows, graph.rowOffsets.data(),
                             (vertexCount + 1) * sizeof(int), cudaMemcpyHostToDevice), "Copying rows");
        if (edgeCount > 0) {
            checkCuda(cudaMemcpy(deviceColumns, graph.columnIndices.data(),
                                 edgeCount * sizeof(int), cudaMemcpyHostToDevice), "Copying columns");
            checkCuda(cudaMemcpy(deviceWeights, graph.weights.data(),
                                 edgeCount * sizeof(int), cudaMemcpyHostToDevice), "Copying weights");
        }

        vector<int> distance(vertexCount, INF);
        distance[source] = 0;
        checkCuda(cudaMemcpy(deviceOldDistance, distance.data(),
                             vertexCount * sizeof(int), cudaMemcpyHostToDevice), "Copying distances");

        const int threadsPerBlock = 256;
        int blocks = (vertexCount + threadsPerBlock - 1) / threadsPerBlock;
        for (int iteration = 0; iteration < vertexCount - 1; ++iteration) {
            checkCuda(cudaMemcpy(deviceNewDistance, deviceOldDistance,
                                 vertexCount * sizeof(int), cudaMemcpyDeviceToDevice),
                      "Copying distances for next iteration");
            checkCuda(cudaMemset(deviceChanged, 0, sizeof(int)), "Resetting changed flag");
            relaxEdges<<<blocks, threadsPerBlock>>>(deviceRows, deviceColumns, deviceWeights,
                                                     vertexCount, deviceOldDistance,
                                                     deviceNewDistance, deviceChanged);
            checkCuda(cudaGetLastError(), "Launching relaxEdges");
            checkCuda(cudaDeviceSynchronize(), "Waiting for relaxEdges");

            int changed;
            checkCuda(cudaMemcpy(&changed, deviceChanged, sizeof(int), cudaMemcpyDeviceToHost),
                      "Copying changed flag");
            swap(deviceOldDistance, deviceNewDistance);
            if (!changed) {
                break;
            }
        }

        checkCuda(cudaMemcpy(distance.data(), deviceOldDistance,
                             vertexCount * sizeof(int), cudaMemcpyDeviceToHost), "Copying results");
        cudaFree(deviceRows);
        cudaFree(deviceColumns);
        cudaFree(deviceWeights);
        cudaFree(deviceOldDistance);
        cudaFree(deviceNewDistance);
        cudaFree(deviceChanged);
        return distance;
    } catch (...) {
        cudaFree(deviceRows);
        cudaFree(deviceColumns);
        cudaFree(deviceWeights);
        cudaFree(deviceOldDistance);
        cudaFree(deviceNewDistance);
        cudaFree(deviceChanged);
        throw;
    }
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

        cout << "GPU Bellman-Ford distances from vertex 0:\n";
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