#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

SEGMENTATION_FILE="/output/images/breast-cancer-segmentation-for-tils/segmentation.tif"
DETECTION_FILE="/output/detected-lymphocytes.json"
TILS_SCORE_FILE="/output/til-score.json"
#BULK_TUMOUR="/output/segmentation.xml"

MEMORY=30g

echo "Building docker"
./build.sh

echo "Creating volume..."
docker volume create tiger-output

mkdir $SCRIPTPATH/results

echo "Running algorithm..."
docker run --rm \
        --memory=$MEMORY \
        --memory-swap=$MEMORY \
        --network=none \
        --cap-drop=ALL \
        --security-opt="no-new-privileges" \
        --shm-size=128m \
        --pids-limit=256 \
        --gpus all \
        --cpus 4 \
        -v $SCRIPTPATH/../testinput/:/input/ \
        -v tiger-output:/output/ \
        tigerexamplealgorithm

echo "Checking output files..."
docker run --rm --name dummy \
        -v tiger-output:/output/ \
        python:3.8-slim /bin/bash -c \
        "[[ -f "$SEGMENTATION_FILE" ]] && printf 'Expected file %s exists\!\n' "$SEGMENTATION_FILE"; \
        [[ -f "$TILS_SCORE_FILE" ]] && printf 'Expected file %s exists\!\n' "$TILS_SCORE_FILE""

echo "Copying output files..."
docker run -d --name dummycp \
        -v tiger-output:/output/ \
        python:3.8-slim
docker cp dummycp:$SEGMENTATION_FILE $SCRIPTPATH/results/segmentation.tif
docker cp dummycp:$DETECTION_FILE $SCRIPTPATH/results/detected-lymphocytes.json
docker cp dummycp:$TILS_SCORE_FILE $SCRIPTPATH/results/til-score.json
#docker cp dummycp:$BULK_TUMOUR $SCRIPTPATH/results/bulk_tumour.xml
docker stop dummycp
docker rm dummycp

echo "Removing volume..."
docker volume rm tiger-output