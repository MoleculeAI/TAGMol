TOTAL_TASKS=100
BATCH_SIZE=100

if [ $# != 2 ]; then
    echo "Error: 2 arguments required."
    exit 1
fi

CONFIG_FILE=$1
RESULT_PATH="experiments/$2"
NODE_ALL=1
NODE_THIS=0
START_IDX=0

for ((i=$START_IDX;i<$TOTAL_TASKS;i++)); do
    NODE_TARGET=$(($i % $NODE_ALL))
    if [ $NODE_TARGET == $NODE_THIS ]; then
        echo "Task ${i} assigned to this worker (${NODE_THIS})"
        python -m scripts.sample_diffusion ${CONFIG_FILE} -i ${i} --batch_size ${BATCH_SIZE} --result_path ${RESULT_PATH}
    fi
done
