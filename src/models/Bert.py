import transformers
import torch

class Model(torch.nn.Module):

    def __init__(
            self,
            vocab_size                   : int   = 30522      ,
            hidden_size                  : int   = 768        ,
            num_hidden_layers            : int   = 12         ,
            num_attention_heads          : int   = 12         ,
            intermediate_size            : int   = 3072       ,
            hidden_act                   : str   = "gelu"     ,
            hidden_dropout_prob          : float = 0.1        ,
            attention_probs_dropout_prob : float = 0.1        ,
            max_position_embeddings      : int   = 512        ,
            type_vocab_size              : int   = 2          ,
            initializer_range            : float = 0.02       ,
            layer_norm_eps               : float = 1e-12      ,
            gradient_checkpointing       : bool  = False      ,
            position_embedding_type      : str   = "absolute" ,
            use_cache                    : bool  = True       ,
            tie_word_embeddings          : bool  = True       ,
        ):
        super(Model, self).__init__()

        self.BertConfig = transformers.BertConfig(
            vocab_size                   = vocab_size                   ,
            hidden_size                  = hidden_size                  ,
            num_hidden_layers            = num_hidden_layers            ,
            num_attention_heads          = num_attention_heads          ,
            intermediate_size            = intermediate_size            ,
            hidden_act                   = hidden_act                   ,
            hidden_dropout_prob          = hidden_dropout_prob          ,
            attention_probs_dropout_prob = attention_probs_dropout_prob ,
            max_position_embeddings      = max_position_embeddings      ,
            type_vocab_size              = type_vocab_size              ,
            initializer_range            = initializer_range            ,
            layer_norm_eps               = layer_norm_eps               ,
            gradient_checkpointing       = gradient_checkpointing       ,
            position_embedding_type      = position_embedding_type      ,
            use_cache                    = use_cache                    ,
            tie_word_embeddings          = tie_word_embeddings          ,
        )
        self.encoder = transformers.BertForMaskedLM(self.BertConfig)

    def forward(self, input_ids, token_type_ids, attention_mask):
        return self.encoder(
            input_ids      = input_ids      ,
            token_type_ids = token_type_ids ,
            attention_mask = attention_mask ,
        ).logits

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.encoder.cls.predictions.decoder

    @torch.no_grad()
    def untie(self):
        embeddings = self.get_input_embeddings().weight.detach()
        self. get_input_embeddings().weight[:] = torch.nn.Parameter(embeddings.clone(), requires_grad=True)
        self.get_output_embeddings().weight[:] = torch.nn.Parameter(embeddings.clone(), requires_grad=True)

    @torch.no_grad()
    def save(self, path:str):
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def load(self, path:str):
        self.load_state_dict(torch.load(path))
