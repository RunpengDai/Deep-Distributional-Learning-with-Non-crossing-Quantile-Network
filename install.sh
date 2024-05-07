conda create -y --name distRL python=3.6.13
conda init bash
conda activate distRL
wget https://www.atarimania.com/roms/Atari-2600-VCS-ROM-Collection.zip
unzip Atari-2600-VCS-ROM-Collection.zip
pip install --upgrade pip setuptools wheel

pip install atari-py
pip install  -r requirements.txt --verbose
python -m atari_py.import_roms ROMS
