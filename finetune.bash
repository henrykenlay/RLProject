./docker/run.sh 0 python project/finetune.py --modelname control-0 --experiment-name control-0 --reinforce-batchsize 32 --reinforce-shuffle
./docker/run.sh 0 python project/finetune.py --modelname control-1 --experiment-name control-1 --reinforce-batchsize 32 --reinforce-shuffle
./docker/run.sh 0 python project/finetune.py --modelname control-2 --experiment-name control-2 --reinforce-batchsize 32 --reinforce-shuffle
./docker/run.sh 0 python project/finetune.py --modelname control-3 --experiment-name control-3 --reinforce-batchsize 32 --reinforce-shuffle
./docker/run.sh 0 python project/finetune.py --modelname control-4 --experiment-name control-4 --reinforce-batchsize 32 --reinforce-shuffle

./docker/run.sh 1 python project/finetune.py --modelname control-0 --experiment-name control-0-H2 --reinforce-batchsize 32 --reinforce-shuffle --H 2
./docker/run.sh 1 python project/finetune.py --modelname control-1 --experiment-name control-1-H2 --reinforce-batchsize 32 --reinforce-shuffle --H 2
./docker/run.sh 1 python project/finetune.py --modelname control-2 --experiment-name control-2-H2 --reinforce-batchsize 32 --reinforce-shuffle --H 2
./docker/run.sh 1 python project/finetune.py --modelname control-3 --experiment-name control-3-H2 --reinforce-batchsize 32 --reinforce-shuffle --H 2
./docker/run.sh 1 python project/finetune.py --modelname control-4 --experiment-name control-4-H2 --reinforce-batchsize 32 --reinforce-shuffle --H 2

./docker/run.sh 2 python project/finetune.py --modelname control-0 --experiment-name control-0-H4 --reinforce-batchsize 32 --reinforce-shuffle --H 4
./docker/run.sh 2 python project/finetune.py --modelname control-1 --experiment-name control-1-H4 --reinforce-batchsize 32 --reinforce-shuffle --H 4
./docker/run.sh 2 python project/finetune.py --modelname control-2 --experiment-name control-2-H4 --reinforce-batchsize 32 --reinforce-shuffle --H 4
./docker/run.sh 2 python project/finetune.py --modelname control-3 --experiment-name control-3-H4 --reinforce-batchsize 32 --reinforce-shuffle --H 4
./docker/run.sh 2 python project/finetune.py --modelname control-4 --experiment-name control-4-H4 --reinforce-batchsize 32 --reinforce-shuffle --H 4

./docker/run.sh 4 python project/finetune.py --modelname control-0 --experiment-name control-0-H1 --reinforce-batchsize 32 --reinforce-shuffle --H 1
./docker/run.sh 4 python project/finetune.py --modelname control-1 --experiment-name control-1-H1 --reinforce-batchsize 32 --reinforce-shuffle --H 1
./docker/run.sh 4 python project/finetune.py --modelname control-2 --experiment-name control-2-H1 --reinforce-batchsize 32 --reinforce-shuffle --H 1
./docker/run.sh 4 python project/finetune.py --modelname control-3 --experiment-name control-3-H1 --reinforce-batchsize 32 --reinforce-shuffle --H 1
./docker/run.sh 4 python project/finetune.py --modelname control-4 --experiment-name control-4-H1 --reinforce-batchsize 32 --reinforce-shuffle --H 1
