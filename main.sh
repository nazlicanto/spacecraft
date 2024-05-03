#!/usr/bin/env bash

DATA_DIR="../data" 
SUBMISSION_DIR="/sub" 
OUTPUT_PATH="${SUBMISSION_DIR}/predictions.csv"

mkdir -p ${SUBMISSION_DIR}

python main.py ${DATA_DIR} ${OUTPUT_PATH}

