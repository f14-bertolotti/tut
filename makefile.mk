DEVICE="cuda:0"

venv/bin/python:
	virtualenv venv
	venv/bin/pip install -r requirements.txt

data/bert-m/lastmodel.pth: venv/bin/python
	mkdir -p $(dir $@)
	venv/bin/python src/train.py \
		--dir $(dir $@) \
		--etc 100000 \
		--etv 1000 \
		--epochs 30 \
		--train-batch-size 128 \
		--valid-batch-size 32 \
		--learning-rate 0.0001 \
		--device $(DEVICE) \
		--compile True \
		--arch "hidden_size" 256 \
		--arch "intermediate_size" 1024 \
		--arch "num_hidden_layers" 4 \
		--arch "num_attention_heads" 2 \
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
		--learning-rate 0.0001 \
		--device $(DEVICE) \
		--compile True \
		--arch "hidden_size" 256 \
		--arch "intermediate_size" 1024 \
		--arch "num_hidden_layers" 4 \
		--arch "num_attention_heads" 2 \
		--arch "tie_word_embeddings" False \
		--restore data/bert-m/ 200000
