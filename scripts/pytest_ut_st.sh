pytest -v -s -c pytest.ini -m 'not download and not gpu_only' tests/ut
pytest tests/st