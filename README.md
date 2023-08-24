# TorchGNN-GCN

Integrated example of GCN functionality in ROOT's TorchGNN.

## Dependencies

- A Python virtual environment with:
    - PyTorch
    - PyTorch Geometric
    - ROOT
- C++
    - LibTorch: Can be downloaded from the [PyTorch homepage](https://pytorch.org/). We used version TorchLib 2.0.1 (
      cxx11 ABI).
    - BLAS: We used [OpenBLAS](https://www.openblas.net/).
    - PyTorch Scatter: See installation instructions at
      the [PyTorch Scatter repository](https://github.com/rusty1s/pytorch_scatter#c-api).
    - PyTorch Sparse: See installation instructions at
      the [PyTorch Sparse repository](https://github.com/rusty1s/pytorch_sparse#c-api).
        - Make sure that the third-party library parallel-hashmap is included in the ```third_party``` directory.

Both PyTorch Scatter and Sparse can be installed in the following way:

```
cd source_code_directory
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="path/to/libtorch" -DCMAKE_BUILD_TYPE="Release" ..
make
make install
```

## How to run

The code can be compiled and run in the following way:

```
cd code_directory
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH="path/to/libtorch" -DCMAKE_BUILD_TYPE="Release" ..
make
export OMP_NUM_THREADS=1
./TorchGNN
```

Alternatively, one can use the provided bash script ```collect_statistics.sh```, to collect all statistics mentioned in
the report. Make sure to activate the virtual environment before calling the bash script.
