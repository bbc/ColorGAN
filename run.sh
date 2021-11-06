#!/bin/sh

help() {
    echo
    echo "\
    Options:
      -h; Show help.
      -a <action>; Select action (train | test).
      -c <path>; Path to config file.
      -g <number>; Cuda device number (default: 0).
      -m <number>; Memory shm-size (G) (default: 8).
    " | column -t -s ";"
}

error() { help; exit 1; }

# Defaults
DEVICE=0
MEMORY=8G
DATA=/work/marcb/data/reduced_imagenet

while getopts "ha:d:c:g:m:" OPTIONS; do
    case $OPTIONS in
        h) help; exit 1;;
        a) ACTION=${OPTARG} ;;
        d) DATA=${OPTARG} ;;
        c) CONFIG=${OPTARG} ;;
        g) DEVICE=${OPTARG} ;;
        m) MEMORY=${OPTARG}G ;;
    esac
done

[ -z $CONFIG ] && { echo "Error: Provide config file."; error; }
# [ -z $DATA ] && { echo "Error: Provide data path."; error; }

case $ACTION in
    train)
        mkdir -p experiments;
        nvidia-docker run -it -e NVIDIA_VISIBLE_DEVICES=$DEVICE \
        --mount "type=bind,src=$(pwd)/config/$CONFIG,dst=/app/config.py" \
        --mount "type=bind,src=$(pwd),dst=/app/src" \
        --mount "type=bind,src=$DATA,dst=/app/data" \
        --mount "type=bind,src=$(pwd)/pretrained,dst=/app/pretrained" \
        --mount "type=bind,src=$(pwd)/experiments,dst=/app/experiments" \
        --user $(id -u):$(id -g) --shm-size $MEMORY colorgan \
        python src/main_test.py -c /app/config.py -g 0 -a train;;
        # bash ;;
esac