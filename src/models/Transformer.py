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



        self.input_embeddings    = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position_embeddings = torch.nn.Embedding(max_position_embeddings, hidden_size)
        self.output_embeddings   = self.input_embeddings.weight if tie_word_embeddings else torch.nn.Parameter(torch.nn.init.normal_(torch.empty(vocab_size, hidden_size), mean=0.0, std=initializer_range), requires_grad=True)

        self.encoder = torch.nn.TransformerEncoder(
            torch.nn.TransformerEncoderLayer(
                d_model          = hidden_size         ,
                nhead            = num_attention_heads ,
                dim_feedforward  = intermediate_size   ,
                dropout          = hidden_dropout_prob ,
                activation       = hidden_act          ,
                layer_norm_eps   = layer_norm_eps      ,
                batch_first      = True                ,
            ),
            num_layers = num_hidden_layers,
        )

        # intialize weights
        self.   input_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        self.  output_embeddings       .data.normal_(mean=0.0, std=initializer_range)
        self.position_embeddings.weight.data.normal_(mean=0.0, std=initializer_range)
        self.   input_embeddings.weight.data[0].zero_()
        self.  output_embeddings       .data[0].zero_()


    def forward(self, input_ids, token_type_ids, attention_mask):
        embeddings = self.input_embeddings(input_ids) + self.position_embeddings.weight[:input_ids.size(1)]
        encoded = self.encoder(embeddings, src_key_padding_mask = attention_mask.bool())
        logits = encoded @ self.output_embeddings.T
        return logits

    def get_input_embeddings(self):
        return self.input_embeddings

    def get_output_embeddings(self):
        return self.output_embeddings

    @torch.no_grad()
    def untie(self):
        input_embeddings  = self.get_input_embeddings().weight.detach()
        output_embeddings = self.get_output_embeddings().weight.detach()
        self. get_input_embeddings().weight[:] = torch.nn.Parameter( input_embeddings.clone(), requires_grad=True)
        self.get_output_embeddings().weight[:] = torch.nn.Parameter(output_embeddings.clone(), requires_grad=True)

    @torch.no_grad()
    def save(self, path:str):
        torch.save(self.state_dict(), path)

    @torch.no_grad()
    def load(self, path:str):
        self.load_state_dict(torch.load(path))
 
