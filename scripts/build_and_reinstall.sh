rm -rf ./dist
python setup.py bdist_wheel
rm -rf *.egg-info
pip uninstall mindhf mindnlp -y && pip install dist/*.whl