conda create -y --name distRL python=3.6.13
conda activate distRL
pip install atari-py
python -m atari_py.import_roms /usr/src/app/Atari
pip install  -r requirements.txt --verbose