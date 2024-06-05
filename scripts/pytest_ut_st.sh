export RUN_SLOW=True
#pylint mindnlp --rcfile=.github/pylint.conf
pytest -v -s -c pytest.ini /data/wjk/mindnlp/tests/ut/transformers/models/nystromformer
#pytest tests/st