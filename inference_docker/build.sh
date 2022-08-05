#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build --platform linux/amd64 -t tigerexamplealgorithm "$SCRIPTPATH"
