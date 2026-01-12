#!/bin/bash

rjob submit \
    --name=0108-spec2smi \
    --gpu=4 \
    --memory=900000 \
    --cpu=32 \
    --charged-group=ma4science_gpu \
    --private-machine=group \
    --mount=gpfs://gpfs1/yangzhuo:/mnt/shared-storage-user/yangzhuo \
    --image=registry.h.pjlab.org.cn/ailab-ai4chem-ai4chem_gpu/yangzhuo-cuda121:20251028101900 \
    -P 1 \
    --host-network=true \
    --custom-resources brainpp.cn/fuse=1 \
    -e DISTRIBUTED_JOB=true \
    -- bash -exc "/mnt/shared-storage-user/yangzhuo/main/projects/slm/Spectra2Smiles-AR/Spectra2Smiles-AR/scripts/start_training.sh"