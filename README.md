# TS-imgnet

This repo implements Teacher-Student for Tarflow on ImageNet latent.

Teacher p2b8l8 160ep noise 0.05 checkpoint: `/kmh-nfs-ssd-eu-mount/logs/sqa/sqa_Flow_matching/20250411_151116_hts34m_kmh-tpuvm-v3-32-1__b_lr_ep_torchvision_r50_eval/checkpoint_800640`

## Setting up the environment

Don't forget to add your `config.sh`: 

```shell
export EU_IMAGENET_FAKE="/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/kmh-nfs-ssd-eu-mount/" # path to the fake imagenet data at EU
export US_IMAGENET_FAKE="/kmh-nfs-ssd-eu-mount/code/hanhong/data/imagenet_fake/kmh-nfs-us-mount/" # path to the fake imagenet data at US
export OWN_CONDA_ENV_NAME=NNX # A useable env at your V3-8
export WANDB_API_KEY= # Your wandb API key
export TASKNAME=fm_linen # Task name
```
