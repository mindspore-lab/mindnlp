export RUN_SLOW=True
pytest -v -s -c pytest.ini tests/ut
pytest tests/st
