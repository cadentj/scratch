# %%

from nnterp import StandardizedTransformer

from nnsight import CONFIG

CONFIG.set_default_api_key("c6281f57-5471-4fce-8ef1-3400cbf6f158")

model = StandardizedTransformer("gpt2")  # or any transformer
with model.trace("Hello", remote=True):
    layer_5_out = model.layers_output[5]
    model.layers_output[10] = layer_5_out  # same API for all models
# %%
