python3 -m venv MyVenv
source ./MyVenv/bin/activate
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip3 install -r requirements.txt
cp ./export_nir.py ./MyVenv/lib/python3.12/site-packages/snntorch/export_nir.py