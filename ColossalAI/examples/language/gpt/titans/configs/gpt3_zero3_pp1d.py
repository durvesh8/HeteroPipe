from model import GPT3_pipeline_hybrid, GPT3_pipeline_hybridgpt2small

from colossalai.nn.optimizer import HybridAdam
from colossalai.zero.legacy.shard_utils import TensorShardStrategy


pipeline_size = 4
BATCH_SIZE = pipeline_size
NUM_EPOCHS = 1
SEQ_LEN = 2048
NUM_MICRO_BATCHES = pipeline_size
HIDDEN_SIZE = 768
TENSOR_SHAPE = (BATCH_SIZE // NUM_MICRO_BATCHES, SEQ_LEN, HIDDEN_SIZE)

# if you do no want zero, just comment out this dictionary
# zero = dict(model_config=dict(tensor_placement_policy='cuda', shard_strategy=TensorShardStrategy()),
#             optimizer_config=dict(initial_scale=2**16))

optimizer = dict(
    type=HybridAdam,
    lr=0.00015,
    weight_decay=1e-2,
)

experimentvar = True

dpranksvar = [[0,1],[2,3]]
ppranksvar = [[0,2],[1,3]]

# model = dict(type=GPT3_pipeline_hybrid, checkpoint=True, num_chunks=1)
model = dict(type=GPT3_pipeline_hybridgpt2small, checkpoint=True, num_chunks=1,num_attention_heads=12,hidden_size=HIDDEN_SIZE,
                            max_position_embeddings=2048,num_layers=12,experiment=experimentvar,dpranks=dpranksvar,ppranks=ppranksvar)
#model = dict(type=GPT3_pipeline_hybridgpt2small, checkpoint=True, num_chunks=1,experiment=experimentvar,dpranks=dpranksvar,ppranks=ppranksvar)


# pipeline parallel: modify integer value for the number of pipeline stages
# tensor parallel: modify size to set the tensor parallel size, usually the number of GPUs per node
# for the current model implementation, mode can only be 1D or None
parallel = dict(
    pipeline=pipeline_size,
    experiment=experimentvar,
    pipeline_ranks=ppranksvar,
    dpranks=dpranksvar,
    tensor=dict(size=1, mode='1d'),    # for the current model implementation, mode can only be 1D or None
)
