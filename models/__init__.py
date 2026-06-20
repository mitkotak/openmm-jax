import os

# Alanine-dipeptide-explicit test goes OOM without this
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
