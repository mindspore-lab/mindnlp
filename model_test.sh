read -p "请输入模型名称: " name
pylint mindnlp --rcfile=.github/pylint.conf
pytest -v -s -c pytest.ini  tests/ut/transformers/models/${name} 

