# ======================================================= #
#                  SETUP SPOTPY-GPRMAX                    #
# ======================================================= #

# load modules
source env2019a.sh

# create a virtual Python environment in the current folder
# and overwrite any existing one
python -m venv --clear venv

# activate the virtual environment
source venv/bin/activate

# install requirements for spotpy-gprMax to the virtual environment
pip install -r requirements.txt

# install spotpy to the virtual environment
cd spotpy || echo "ERROR: folder spotpy does not exist!"; exit
python setup.py install
cd ..

# install gprMax to the virtual environment
cd gprMax || echo "ERROR: folder gprMax does not exist!"; exit
python setup.py build
python setup.py install
cd ..
