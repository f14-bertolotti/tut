DEVICE="cuda:0"

EPOCHS=30
ETC=100000
COMPILER=basic
DATASIZE=None
RESTORE=""
TIED=False
IOEMB=False

venv/bin/python:
	virtualenv venv
	venv/bin/pip install -r requirements.txt

data/%/final.pt: venv/bin/python
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc $(ETC) \
		--etv 1000 \
		--epochs $(EPOCHS) \
		--train-batch-size 128 \
		--valid-batch-size 64 \
		--device $(DEVICE) \
		--compiler $(COMPILER) \
		--ioemb-copy $(IOEMB) \
		--data "name" "Wikitext" \
		--data "dataset_size" $(DATASIZE) \
		--opti "name" "AdamW" \
		--opti "lr" 0.0005 \
		--opti "weight_decay" 0.01 \
		--arch "name" "Bert" \
		--arch "hidden_size" 512 \
		--arch "intermediate_size" 2048 \
		--arch "num_hidden_layers" 3 \
		--arch "num_attention_heads" 4 \
		--arch "initializer_range" 0.02 \
		--arch "tie_word_embeddings" $(TIED) \
		--restore $(RESTORE)
