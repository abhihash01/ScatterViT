# ScatterViT
End-to-end foundation-model pipeline for X-Ray scattering Images from Synchrotron—Self Supervised pre-training, parameter-efficient fine-tuning, INT8 deployment. Fine tuned for downstream classification. 

##check

#!/bin/bash

xray_scattering_fm/
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   ├── synthetic/
│   └── external/
│
├── scripts/
│   ├── data_preprocessing.py
│   ├── synthetic_generation.py
│   ├── split_dataset.py
│   └── utils.py
│
├── models/
│   ├── dinov2/
│   │   ├── ssl_pretrain.py
│   │   ├── lora_finetune.py
│   │   ├── model_utils.py
│   │   └── checkpoints/
│   ├── resnet34/
│   │   ├── train_baseline.py
│   │   ├── eval_baseline.py
│   │   └── checkpoints/
│   ├── diffusion/
│   ├── mae/
│   └── sam/
│
├── training/
│   ├── distributed/
│   └── wandb/
│
├── configs/
│   ├── model/
│   ├── training/
│   └── data/
│
├── evaluation/
│   ├── metrics.py
│   ├── evaluate_ssl.py
│   └── evaluate_baseline.py
│
├── deployment/
│   ├── inference_pipeline.py
│   ├── quantization.py
│   ├── active_learning.py
│   └── docker/
│
├── notebooks/
│   ├── exploratory/
│   ├── experiments/
│   └── deployment/
│
├── logs/
├── tests/
│
├── requirements.txt
├── README.md
└── .gitignore

