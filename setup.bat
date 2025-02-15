@echo off

:: Create the virtual environment
:: Delete virtual environment folder if it exists
rmdir /S /Q myenv

:: Create and activate virtual environment
py -3.10 -m venv myenv
call myenv\Scripts\activate.bat

pip install jupyter ipykernel
pip install --no-cache-dir -r requirements.txt
python -m ipykernel install --user --name=myenv --display-name "Python (myenv)"

:: Launch Jupyter Notebook
jupyter-notebook dCMF.ipynb