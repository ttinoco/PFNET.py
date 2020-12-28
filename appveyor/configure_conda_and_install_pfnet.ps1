conda create -n testenv --yes python=$env:PYTHON_VERSION pip
conda activate testenv
"python=$env:PYTHON_VERSION" | Out-File C:\conda\envs\testenv\conda-meta\pinned -encoding ascii
pip install -r requirements.txt
