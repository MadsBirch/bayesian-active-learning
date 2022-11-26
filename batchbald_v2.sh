#!/bin/bash

JOB_NAME=bank_marketing_$(date +%Y%m%d_%H%M%S)
REGION=europe-west1
BUCKET_NAME=bal-bucket

gcloud ai-platform jobs submit training ${JOB_NAME} \
    --region=$REGION \
    --master-image-uri=gcr.io/bayesian-al/batchbald:latest \
    --scale-tier=CUSTOM \
    --master-machine-type=n1-standard-8 \
    --master-accelerator=type=nvidia-tesla-k80,count=1 \
    -- \
    --storage-path=gs://$BUCKET_NAME/$JOB_NAME \