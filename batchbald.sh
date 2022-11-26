#!/bin/bash

JOB_NAME=batchbald_$(date +%Y%m%d_%H%M%S)
REGION=europe-west1
BUCKET_NAME=bal-bucket

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=$REGION \
    --master-image-uri=gcr.io/bayesian-al/batchbald:latest \