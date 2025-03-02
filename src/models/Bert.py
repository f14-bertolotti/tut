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
            hidden_dropout_prob          : float = 0.0        ,
            max_position_embeddings      : int   = 512        ,
            initializer_range            : float = 0.02       ,
            layer_norm_eps               : float = 1e-12      ,
            tie_word_embeddings          : bool  = True       ,
        ):
        super(Model, self).__init__()

        self. input_embeddings = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.   pos_embeddings = torch.nn.Parameter(torch.empty(max_position_embeddings, hidden_size), requires_grad=True)
        self.output_embeddings = torch.nn.Parameter(torch.empty(vocab_size, hidden_size), requires_grad=True) 
        self.layer_norm = torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.tied = tie_word_embeddings

        self.encoder = transformers.BertModel(
            transformers.BertConfig(
                vocab_size              = vocab_size              , 
                hidden_size             = hidden_size             , 
                num_hidden_layers       = num_hidden_layers       , 
                num_attention_heads     = num_attention_heads     , 
                intermediate_size       = intermediate_size       , 
                hidden_act              = hidden_act              , 
                hidden_dropout_prob     = hidden_dropout_prob     , 
                max_position_embeddings = max_position_embeddings ,
                initializer_range       = initializer_range       , 
                layer_norm_eps          = layer_norm_eps          , 
            )
        )
 
        # intialize embedding weights
        with torch.no_grad():
            self. input_embeddings.weight.normal_(mean=0.0, std=initializer_range)
            self.output_embeddings       .normal_(mean=0.0, std=initializer_range)
            self.   pos_embeddings       .normal_(mean=0.0, std=initializer_range)
            self. input_embeddings.weight[0].zero_()
            self.output_embeddings       [0].zero_()

    def forward(self, input_ids, attention_mask):
        embeddings = self.layer_norm(self.input_embeddings(input_ids) + self.pos_embeddings[:input_ids.size(1)])
        encoded = self.encoder(inputs_embeds=embeddings, attention_mask = attention_mask.bool())
        logits = encoded.last_hidden_state @ self.get_output_embeddings().T
        return logits

    def get_input_embeddings(self):
        return self.input_embeddings.weight

    def get_output_embeddings(self):
        return self.input_embeddings.weight if self.tied else self.output_embeddings

