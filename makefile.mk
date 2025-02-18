DEVICE="cuda:0"

venv/bin/python:
	virtualenv venv
	venv/bin/pip install -r requirements.txt


# Trains a super tiny BERT for testing purposes
data/bert-xxs/lastmodel.pth: venv/bin/python
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc 10000 \
		--etv 10000 \
		--epochs 2 \
		--train-batch-size 128 \
		--valid-batch-size 32 \
		--learning-rate 0.01 \
		--device $(DEVICE) \
		--arch "hidden_size" 10 \
		--arch "intermediate_size" 10 \
		--arch "num_hidden_layers" 1 \
		--arch "num_attention_heads" 1 \
		--arch "tie_word_embeddings" True
