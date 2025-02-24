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
        
        self.output_embeddings = self.get_input_embeddings().weight if tie_word_embeddings else \
                                 torch.nn.Parameter(torch.empty(vocab_size,hidden_size).normal_(mean=0.0, std=initializer_range), requires_grad=True)

    def forward(self, input_ids, token_type_ids, attention_mask):
        encoded = self.encoder(input_ids=input_ids, attention_mask = attention_mask.bool())
        logits = encoded.last_hidden_state @ self.output_embeddings.T
        return logits

    def get_input_embeddings(self):
        return self.encoder.get_input_embeddings()

    def get_output_embeddings(self):
        return self.output_embeddings

    @torch.no_grad()
    def untie(self):
        input_embeddings  = self. get_input_embeddings().weight.detach()
        output_embeddings = self.get_output_embeddings().detach()
        self. get_input_embeddings().weight[:] = torch.nn.Parameter( input_embeddings.clone(), requires_grad=True)
        self.get_output_embeddings()       [:] = torch.nn.Parameter(output_embeddings.clone(), requires_grad=True)

    @torch.no_grad()
    def save(self, path:str):
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def load(self, path:str):
        self.load_state_dict(torch.load(path))
 
