python setup.py bdist_wheel
pip uninstall mindnlp -y && pip install --force-reinstall dist/*.whl