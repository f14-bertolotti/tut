import transformers
import torch

class Model(torch.nn.Module):

    def __init__(
            self,
        ):
        super(Model, self).__init__()

        self.BertConfig = transformers.BertConfig(
            vocab_size = 30522,
            hidden_size = 768,
            num_hidden_layers = 12,
            num_attention_heads = 12,
            intermediate_size = 3072,
            hidden_act = "gelu",
            hidden_dropout_prob = 0.1,
            attention_probs_dropout_prob = 0.1,
            max_position_embeddings = 512,
            type_vocab_size = 2,
            initializer_range = 0.02,
            layer_norm_eps = 1e-12,
            gradient_checkpointing = False,
            position_embedding_type = "absolute",
            use_cache = True,
        )
        self.BertModel = transformers.BertForMaskedLM(self.BertConfig)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.BertModel(
            input_ids      = input_ids      ,
            token_type_ids = token_type_ids ,
            attention_mask = attention_mask ,
        ).logits




