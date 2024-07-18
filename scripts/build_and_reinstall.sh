rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall mindnlp -y && pip install dist/*.whl