DEVICE="cuda:0"

venv/bin/python:
	virtualenv venv
	venv/bin/pip install -r requirements.txt

data/transformer/lastmodel.pth: venv/bin/python
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc 100000 \
		--etv 1000 \
		--epochs 3000 \
		--train-batch-size 128 \
		--valid-batch-size 64 \
		--grad-acc-steps 1 \
		--device $(DEVICE) \
		--compile False \
		--data "name" "Wikitext" \
		--opti "name" "AdamW" \
		--opti "lr" 0.001 \
		--opti "weight_decay" 0.01 \
		--arch "name" "Transformer" \
		--arch "hidden_size" 128 \
		--arch "intermediate_size" 512 \
		--arch "num_hidden_layers" 4 \
		--arch "num_attention_heads" 2 \
		--arch "initializer_range" 0.02 \
		--arch "tie_word_embeddings" True

data/bert-m/untied200k/lastmodel.pth: venv/bin/python
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc 100000 \
		--etv 1000 \
		--epochs 30 \
		--train-batch-size 128 \
		--valid-batch-size 32 \
		--device $(DEVICE) \
		--compile True \
		--opti "lr" 0.0001 \
		--opti "weight_decay" 0.01 \
		--arch "hidden_size" 256 \
		--arch "intermediate_size" 1024 \
		--arch "num_hidden_layers" 4 \
		--arch "num_attention_heads" 2 \
		--arch "tie_word_embeddings" False \
		--restore data/bert-m/ 200000

all: data/bert-m/lastmodel.pth data/bert-m/untied200k/lastmodel.pth
