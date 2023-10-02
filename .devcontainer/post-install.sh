poetry config virtualenvs.options.system-site-packages true

poetry update
poetry lock
poetry install --with style