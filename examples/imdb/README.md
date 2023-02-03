# FlaxNLP with IMDB

Setup:

```bash
python -m virtualenv .venv
source .venv/bin/activate
pip insatll -U requirements.txt
```

Configure model/training settings:

```bash
vim config.json
```

Train model:

```bash
python train.py
```

Evaluate trained model:

```bash
python eval.py --subset test
```
