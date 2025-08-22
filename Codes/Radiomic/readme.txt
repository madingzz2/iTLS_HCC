conda create -n radiomic python=3.9 -y
conda activate radiomic

pip install -q pyradiomics SimpleITK nibabel pandas tqdm scikit-learn

# cmd
D:
cd D
python extract_radiomics.py -d /xyh -o xyh_Rad_features.csv -j 6