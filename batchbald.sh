#!/bin/bash

BUCKET_NAME=bal-bucket
JOB_NAME=assert_cuda
JOB_DIR=gs://${BUCKET_NAME}/output/

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west1 \
    --master-image-uri=gcr.io/bayesian-al/batchbald:latest \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-8 \
    --master-accelerator=type=nvidia-tesla-k80,count=1 \