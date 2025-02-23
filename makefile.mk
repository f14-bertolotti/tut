DEVICE="cuda:0"

MODEL=Transformer
EPOCHS=30
MIDPOINT=100000

venv/bin/python:
	virtualenv venv
	venv/bin/pip install -r requirements.txt

data/$(MODEL)/lastmodel.pth: venv/bin/python
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc $(MIDPOINT) \
		--etv 1000 \
		--epochs $(EPOCHS) \
		--train-batch-size 128 \
		--valid-batch-size 64 \
		--grad-acc-steps 1 \
		--device $(DEVICE) \
		--compile True \
		--data "name" "Wikitext" \
		--opti "name" "AdamW" \
		--opti "lr" 0.001 \
		--opti "weight_decay" 0.01 \
		--arch "name" $(MODEL) \
		--arch "hidden_size" 512 \
		--arch "intermediate_size" 2048 \
		--arch "num_hidden_layers" 4 \
		--arch "num_attention_heads" 4 \
		--arch "initializer_range" 0.02 \
		--arch "tie_word_embeddings" True

data/$(MODEL)/mid/lastmodel.pth: data/$(MODEL)/lastmodel.pth
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc $(MIDPOINT) \
		--etv 1000 \
		--epochs $(EPOCHS) \
		--train-batch-size 128 \
		--valid-batch-size 64 \
		--grad-acc-steps 1 \
		--device $(DEVICE) \
		--compile True \
		--data "name" "Wikitext" \
		--opti "name" "AdamW" \
		--opti "lr" 0.001 \
		--opti "weight_decay" 0.01 \
		--arch "name" $(MODEL) \
		--arch "hidden_size" 512 \
		--arch "intermediate_size" 2048 \
		--arch "num_hidden_layers" 4 \
		--arch "num_attention_heads" 4 \
		--arch "initializer_range" 0.02 \
		--arch "tie_word_embeddings" False \
		--restore data/$(MODEL)/$(MIDPOINT)
