# Tutorial 3: Matrix Multiplication using Tensor Cores

## Task
Implement matrix multiplication using **Tensor Cores** for two square matrices of dimensions **64×64**.

- Tensor core should do multiplication as tiles of **16×16**
- A 64×64 matrix will have **16 tiles** (4×4 grid of 16×16 tiles)
- Reference code from class: [gem-tc.cu](https://github.com/unnikrishnan-c/GPUProgramming-2026/blob/main/gem-tc.cu)
