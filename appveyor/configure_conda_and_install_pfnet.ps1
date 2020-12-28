conda create -n testenv --yes python=$env:PYTHON_VERSION pip
conda activate testenv
"python=$env:PYTHON_VERSION" | Out-File C:\conda\envs\testenv\conda-meta\pinned -encoding ascii
conda install -c anaconda --yes libpython
conda install -c msys2 --yes m2w64-toolchain
pip install -r requirements.txt
