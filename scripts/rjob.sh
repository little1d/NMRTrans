#!/bin/bash

rjob submit \
    --name=nmrtrans \
    --gpu=2 \
    --memory=800000 \
    --cpu=16 \
    --charged-group=ai4chem_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/yangzhuo:/mnt/shared-storage-user/yangzhuo \
    --image=registry.h.pjlab.org.cn/ailab-ai4chem-ai4chem_gpu/yangzhuo-cuda121:20251028101900 \
    -P 1 \
    --host-network=true \
    --custom-resources brainpp.cn/fuse=1 \
    -e DISTRIBUTED_JOB=true \
    -- bash -exc "/mnt/shared-storage-user/yangzhuo/main/projects/NMRTrans/scripts/start_training.sh"