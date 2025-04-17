# initialize and set up remote TPU VM
source ka.sh # import VM_NAME, ZONE

echo $VM_NAME $ZONE


gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
--worker=all --command "
conda activate NNX
pip3 install flax==0.10.2
pip3 show flax
python3 -c 'import flax.nnx as nn; print(nn.Linear)'
"
# python3 -c 'import flax; print(flax.nnx)'

# pip install wandb

