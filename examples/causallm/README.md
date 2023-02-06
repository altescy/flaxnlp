# Causal Language Model Example

Setup:

```bash
python -m virtualenv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Prepare dataset by the following command.
By default, synthetic dataset will be used.

```bash
python prepare.py
```

If you want to use your custom dataset, `--from-file` option is available:

```bash
python prepare.py --from-file path/to/dataset.txt
```

Configure model and training settings:

```bash
vim config.json
```

Train model:

```bash
python train.py
```

Generate text interactively:

```bash
python sample.py
```
