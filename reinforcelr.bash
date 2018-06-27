./docker/run.sh 0 python project/train-model.py --experiment-name reinforce-lr-0.00001 --num-epochs 0 --reinforce --agg-iters 10000 --traj-per-agg 1 --reinforce-lr 0.00001
./docker/run.sh 1 python project/train-model.py --experiment-name reinforce-lr-0.000001 --num-epochs 0 --reinforce --agg-iters 10000 --traj-per-agg 1 --reinforce-lr 0.000001
./docker/run.sh 2 python project/train-model.py --experiment-name reinforce-lr-0.0000001 --num-epochs 0 --reinforce --agg-iters 10000 --traj-per-agg 1 --reinforce-lr 0.0000001
./docker/run.sh 3 python project/train-model.py --experiment-name reinforce-lr-0.0001 --num-epochs 0 --reinforce --agg-iters 10000 --traj-per-agg 1 --reinforce-lr 0.0001
./docker/run.sh 4 python project/train-model.py --experiment-name reinforce-lr-0.001 --num-epochs 0 --reinforce --agg-iters 10000 --traj-per-agg 1 --reinforce-lr 0.001
