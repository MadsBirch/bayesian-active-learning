#!/bin/bash

BUCKET_NAME=bal-bucket
JOB_NAME=job_1
JOB_DIR=gs://${BUCKET_NAME}/${JOB_NAME}/models

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=europe-west1 \
    --master-image-uri=gcr.io/bayesian-al/batchbald:latest \
    --job-dir=${JOB_DIR} \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-8 \
    --master-accelerator=type=nvidia-tesla-k80,count=1 \