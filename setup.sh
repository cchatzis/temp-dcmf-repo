#!/bin/bash

# Delete virtual environment folder if it exists
if [ -d "myenv" ]; then
    rm -rf myenv
fi

#chmod
chmod u+rwx $(pwd)

# Create and activate virtual environment
python3.10 -m venv myenv
source myenv/bin/activate

# Install dependencies
pip install --no-cache-dir -r requirements.txt

pip install jupyter ipykernel notebook
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

# Launch Jupyter Notebook
jupyter-notebook --NotebookApp.use_redirect_file=False dCMF.ipynb