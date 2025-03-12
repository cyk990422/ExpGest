

source activate lipe_env


cd /apdcephfs/private_ykcheng/ICME2024_ExpGest_Demo/ExpGest_mirror

pip install lmdb -i https://mirrors.tencent.com/pypi/simple/
pip install timm -i https://mirrors.tencent.com/pypi/simple/
pip install wandb -i https://mirrors.tencent.com/pypi/simple/
pip install IPython -i https://mirrors.tencent.com/pypi/simple/
pip install librosa -i https://mirrors.tencent.com/pypi/simple/
pip install pyarrow==10.0.0 -i https://mirrors.tencent.com/pypi/simple/
pip install easydict -i https://mirrors.tencent.com/pypi/simple/
pip install configargparse -i https://mirrors.tencent.com/pypi/simple/
pip install einops -i https://mirrors.tencent.com/pypi/simple/
pip install omegaconf -i https://mirrors.tencent.com/pypi/simple/
pip install transformers -i https://mirrors.tencent.com/pypi/simple/
pip install ftfy -i https://mirrors.tencent.com/pypi/simple/
pip install regex -i https://mirrors.tencent.com/pypi/simple/
pip install blobfile -i https://mirrors.tencent.com/pypi/simple/
pip install h5py -i https://mirrors.tencent.com/pypi/simple/
pip install pandas -i https://mirrors.tencent.com/pypi/simple/
cd ../CLIP/
python setup.py install -i https://mirrors.tencent.com/pypi/simple/
pip install --upgrade pytorch torchvision


CUDA_VISIBLE_DEVICES=0  python  test.py


