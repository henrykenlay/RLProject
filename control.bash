# control

#./docker/run.sh 0 python project/train-model.py --experiment-name control-0 --agg-iters 20
#./docker/run.sh 1 python project/train-model.py --experiment-name control-1 --agg-iters 20
#./docker/run.sh 2 python project/train-model.py --experiment-name control-2 --agg-iters 20
#./docker/run.sh 3 python project/train-model.py --experiment-name control-3 --agg-iters 20
#./docker/run.sh 4 python project/train-model.py --experiment-name control-4 --agg-iters 20

# ablation H

#./docker/run.sh 1 python project/train-model.py --experiment-name H5-0 --agg-iters 20 --H 5
#./docker/run.sh 1 python project/train-model.py --experiment-name H5-1 --agg-iters 20 --H 5
#./docker/run.sh 1 python project/train-model.py --experiment-name H5-2 --agg-iters 20 --H 5
#./docker/run.sh 1 python project/train-model.py --experiment-name H5-3 --agg-iters 20 --H 5
#./docker/run.sh 1 python project/train-model.py --experiment-name H5-4 --agg-iters 20 --H 5

#./docker/run.sh 1 python project/train-model.py --experiment-name H10-0 --agg-iters 20 --H 10
#./docker/run.sh 1 python project/train-model.py --experiment-name H10-1 --agg-iters 20 --H 10
#./docker/run.sh 1 python project/train-model.py --experiment-name H10-2 --agg-iters 20 --H 10
#./docker/run.sh 1 python project/train-model.py --experiment-name H10-3 --agg-iters 20 --H 10
#./docker/run.sh 1 python project/train-model.py --experiment-name H10-4 --agg-iters 20 --H 10

# ablation K

#./docker/run.sh 2 python project/train-model.py --experiment-name K250-0 --agg-iters 20 --K 250
#./docker/run.sh 2 python project/train-model.py --experiment-name K250-1 --agg-iters 20 --K 250
#./docker/run.sh 2 python project/train-model.py --experiment-name K250-2 --agg-iters 20 --K 250
#./docker/run.sh 2 python project/train-model.py --experiment-name K250-3 --agg-iters 20 --K 250
#./docker/run.sh 2 python project/train-model.py --experiment-name K250-4 --agg-iters 20 --K 250

./docker/run.sh 0 python project/train-model.py --experiment-name K500-0 --agg-iters 20 --K 500
./docker/run.sh 0 python project/train-model.py --experiment-name K500-1 --agg-iters 20 --K 500
./docker/run.sh 0 python project/train-model.py --experiment-name K500-2 --agg-iters 20 --K 500
./docker/run.sh 0 python project/train-model.py --experiment-name K500-3 --agg-iters 20 --K 500
./docker/run.sh 0 python project/train-model.py --experiment-name K500-4 --agg-iters 20 --K 500

# ablation nn

#./docker/run.sh 4 python project/train-model.py --experiment-name hu125-0 --agg-iters 20 --hidden-units 125
#./docker/run.sh 4 python project/train-model.py --experiment-name hu125-1 --agg-iters 20 --hidden-units 125
#./docker/run.sh 4 python project/train-model.py --experiment-name hu125-2 --agg-iters 20 --hidden-units 125
#./docker/run.sh 4 python project/train-model.py --experiment-name hu125-3 --agg-iters 20 --hidden-units 125
#./docker/run.sh 4 python project/train-model.py --experiment-name hu125-4 --agg-iters 20 --hidden-units 125

./docker/run.sh 1 python project/train-model.py --experiment-name hu250-0 --agg-iters 20 --hidden-units 250
./docker/run.sh 1 python project/train-model.py --experiment-name hu250-1 --agg-iters 20 --hidden-units 250
./docker/run.sh 1 python project/train-model.py --experiment-name hu250-2 --agg-iters 20 --hidden-units 250
./docker/run.sh 1 python project/train-model.py --experiment-name hu250-3 --agg-iters 20 --hidden-units 250
./docker/run.sh 1 python project/train-model.py --experiment-name hu250-4 --agg-iters 20 --hidden-units 250
