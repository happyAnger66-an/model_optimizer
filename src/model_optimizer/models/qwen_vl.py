import torch


class QwenVLWrapper(torch.nn.Module):
    def __init__(self, config, llm, **kwargs):
        super().__init__(**kwargs)
        self.config = config
        self.llm = llm
        self.llm.config._attn_implementation = "eager"

    def forward(self, input_ids, attention_mask, position_ids):
        print(f'here called QwenVLWrapper forward')
        prefix_output = self.llm(input_ids=input_ids,
                                 attention_mask=attention_mask,
                                 position_ids=position_ids,
                                 use_cache=True)
        k_v_caches = []
#        print(f'prefix_output.past_key_values {prefix_output.past_key_values}')
        if prefix_output.past_key_values is not None:
            for keys, values in prefix_output.past_key_values:
                print(f'keys {keys.shape} values {values.shape}')
                k_v_caches.append((keys, values))

        print(f'return ')
        return k_v_caches, prefix_output.last_hidden_state
        #return k_v_caches
