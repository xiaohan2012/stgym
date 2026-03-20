#! /bin/bash

rsync -avzh --compress-level 9 --progress \
    --exclude='*/processed/***' \
    --exclude='gastric-bladder-cancer/raw/GSE246011_extracted/***' \
    --include='brca/***' \
    --include='brca-ptnm-m/***' \
    --include='human-crc/***' \
    --include='mouse-spleen/***' \
    --include='mouse-preoptic/***' \
    --include='human-intestine/***' \
    --include='human-lung/***' \
    --include='breast-cancer/***' \
    --include='mouse-kidney/***' \
    --include='cellcontrast-breast/***' \
    --include='colorectal-cancer/***' \
    --include='upmc/***' \
    --include='charville/***' \
    --include='brca-grade/***' \
    --include='glioblastoma/***' \
    --include='spatial-vdj/***' \
    --include='human-pancreas/***' \
    --include='gastric-bladder-cancer/***' \
    --include='inflammatory-skin/***' \
    --exclude='*' \
    cyy2:stgym/data/ data/
