~~~
#Install 
conda create -n resnet50 python=3.10 -y
conda activate resnet50
nvidia-smi #cuda 12.0 install correct version
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118

#Other dependents
pip install -r requirements.txt

#prepare
python preprocess.py --root /data/raw --out /data/preprocessed --modal AP PVP T2WI

#train
python train.py --modal AP --root /data/preprocessed --seed 42 --epochs 50 --lr 0.01

#feature-extraction
python extract_features.py --modal AP --root /data/preprocessed --out features --dim 1500
~~~