#! /bin/bash

cd ~/TorchGNN-GCN
mkdir ./timings

for RUN in {1..30}
do
	export OMP_NUM_THREADS=16
	python3 GNN_generator.py
	cd build
	cmake -DCMAKE_PREFIX_PATH="/usr/local/include/**;~/libtorch;~/libtorch/**" -DCMAKE_BUILD_TYPE="Release" ..
	make -j16
	export OMP_NUM_THREADS=1
	./TorchGNN
	cd ..
	mv ./timings.csv ./timings/${RUN}.csv
done
python3 statistics_collector.py
