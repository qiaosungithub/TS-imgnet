# Your configurations here
source config.sh
CONDA_ENV=$OWN_CONDA_ENV_NAME

############# No need to modify #############
source ka.sh

echo Running at $VM_NAME $ZONE

STAGEDIR=/$DATA_ROOT/staging/$(whoami)/debug/wandb
sudo mkdir -p $STAGEDIR
sudo chmod 777 $STAGEDIR
echo 'Staging files...'
sudo rsync -a test_wandb.py $STAGEDIR
echo 'staging dir: '$STAGEDIR
echo 'Done staging.'

LOGDIR=$STAGEDIR/log
sudo rm -rf $LOGDIR
sudo mkdir -p ${LOGDIR}
sudo chmod 777 ${LOGDIR}
echo 'Log dir: '$LOGDIR


if [[ $USE_CONDA == 1 ]]; then
    CONDA_PATH=$(which conda)
    CONDA_INIT_SH_PATH=$(dirname $CONDA_PATH)/../etc/profile.d/conda.sh
fi
############# No need to modify [END] #############


################# RUNNING CONFIGS #################
gcloud compute tpus tpu-vm ssh $VM_NAME --zone $ZONE \
    --worker=all --command "
cd $STAGEDIR
echo 'Current dir: '
pwd
if [ \"$USE_CONDA\" -eq 1 ]; then
    echo 'Using conda'
    source $CONDA_INIT_SH_PATH
    conda activate $CONDA_ENV
fi
which python
which pip3
python3 -m wandb login
python3 test_wandb.py
"