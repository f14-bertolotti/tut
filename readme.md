# TUT: Tying Un-Tying

This is the repository to test the performance of LLM technique that involves starting training with tied embeddings and then untying them in the middle of the training. Unfortunately, the preliminary results indicate extremely small gains in performance. This was meant to become a research paper, however the preliminary results are not promising. Since, I am big-believer in publishing also negative result will share this idea as blog [post](https://f14-bertolotti.github.io/posts/06-03-25-tut/index.html). 

## Requirements

* Python 3
* `venv`
* Packages listed in `requirements.txt`:
    * `numpy`
    * `torch`
    * `datasets`
    * `transformers`
    * `click`
    * `tqdm`

## Setup

1.  **Create and activate a virtual environment:**

    ```bash
    make venv/bin/python
    source venv/bin/activate
    ```

    This command will create a virtual environment named `venv` and install the required packages.

## Training

The training process is managed by a Makefile. To start training, run the following command:

```bash
make data/model/final.pt
```

This command will:

* Create the necessary directories.
* Execute the `src/train.py` script with the specified parameters from the Makefile.
* Save checkpoints and the final trained model to the `data/model/` directory.

### Makefile Variables

The Makefile uses the following variables to configure the training process:

* `DEVICE`: The device to use for training (e.g., `cuda:0`, `cpu`).
* `EPOCHS`: The number of training epochs.
* `ETC`: Epochs to wait before saving a checkpoint.
* `COMPILER`: The compiler to use for the model (e.g., `basic`).
* `DATASIZE`: The size of the dataset to use (e.g., `None`).
* `RESTORE`: Path to a checkpoint to restore from.
* `TIED`: Whether to tie word embeddings.
* `IOEMB`: Whether to copy input to output embeddings.

You can modify these variables in the Makefile to customize the training.

### `src/train.py` Script

The `src/train.py` script uses `click` to define command-line options. Here's a brief overview of the key options:

* `--seed`: Random seed for reproducibility.
* `--train-batch-size`, `--valid-batch-size`, `--test-batch-size`: Batch sizes for training, validation, and testing.
* `--epochs`: Number of training epochs.
* `--device`: Device to train on (e.g., `cuda`, `cpu`).
* `--compiler`: Compile the model.
* `--dir`: Directory to save data.
* `--etc`: Epochs to wait before saving.
* `--etv`: Epochs to wait before validating.
* `--restore`: Path to restore from.
* `--ioemb-copy`: Copy input to output embeddings.
* `--arch`, `--opti`, `--data`: Model architecture, optimizer, and dataset parameters.

### Output

The training script generates the following output:

* Checkpoints: Saved periodically during training.
* Final model: `data/model/final.pt`.
* Log files: `train.jsonl`, `valid.jsonl`, `test.jsonl` containing training, validation, and testing metrics.
* state.json: state of the run.


