import sys
import os
import numpy as np
import textwrap
wrapper = textwrap.TextWrapper(width=70)
import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.supervised import training
from argparse import ArgumentParser
from utils import *
from transformer import *

def Encoder(d_model, d_ff, n_heads, dropout, mode, ff_activation):
    """Returns a list of layers that implements a Transformer decoder block.

    The input is an activation tensor.

    Args:
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        mode (str): 'train' or 'eval'.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        list: list of trax.layers.combinators.Serial that maps an activation tensor to an activation tensor.
    """
    causal_attention = tl.CausalAttention(d_model, n_heads=n_heads, dropout=dropout, mode=mode)
    feed_forward = [
        tl.LayerNorm(),
        tl.Dense(d_ff),
        ff_activation(),
        tl.Dropout(rate=dropout, mode=mode),
        tl.Dense(d_model),
        tl.Dropout(rate=dropout, mode=mode)
    ]
    return [
        tl.Residual(tl.LayerNorm(), causal_attention, tl.Dropout(rate=dropout, mode=mode)),
        tl.Residual(feed_forward),
    ]

def TransformerLM(vocab_size=33300,
                  d_model=512,
                  d_ff=2048,
                  n_layers=6,
                  n_heads=8,
                  dropout=0.1,
                  max_len=4096,
                  mode='train',
                  ff_activation=tl.Relu):
    """Returns a Transformer language model.

    The input to the model is a tensor of tokens. (This model uses only the
    decoder part of the overall Transformer.)

    Args:
        vocab_size (int): vocab size.
        d_model (int):  depth of embedding.
        d_ff (int): depth of feed-forward layer.
        n_layers (int): number of decoder layers.
        n_heads (int): number of attention heads.
        dropout (float): dropout rate (how much to drop out).
        max_len (int): maximum symbol length for positional encoding.
        mode (str): 'train', 'eval' or 'predict', predict mode is for fast inference.
        ff_activation (function): the non-linearity in feed-forward layer.

    Returns:
        trax.layers.combinators.Serial: A Transformer language model as a layer that maps from a tensor of tokens
nnn        to activations over a vocab set.
    """

    # Embedding inputs and positional encoder
    positional_encoder = [
        tl.Embedding(vocab_size, d_model),
        tl.Dropout(rate=dropout, mode=mode),
        tl.PositionalEncoding(max_len=max_len, mode=mode)]
    encoder_blocks = [ Encoder(d_model, d_ff, n_heads, dropout, mode, ff_activation) for _ in range(n_layers) ]
    return tl.Serial(
        tl.ShiftRight(mode=mode),
        positional_encoder,
        encoder_blocks,
        tl.LayerNorm(),
        tl.Dense(vocab_size),
        tl.LogSoftmax()
    )

def training_loop(TransformerLM, train_gen, eval_gen, output_dir = "~/model",
                  d_model=512, d_ff=2048, n_layers=6, n_heads=8):
    """
    Input:
ls        TransformerLM (trax.layers.combinators.Serial): The model you are building.
        train_gen (generator): Training stream of data.
        eval_gen (generator): Evaluation stream of data.
        output_dir (str): folder to save your file.

    Returns:
        trax.supervised.training.Loop: Training loop.
    """
    output_dir = os.path.expanduser(output_dir)  # trainer is an object
    lr_schedule = trax.lr.warmup_and_rsqrt_decay(n_warmup_steps=1000, max_value=0.01)

    train_task = training.TrainTask(
        labeled_data=train_gen,
        loss_layer=tl.CrossEntropyLoss(), # Loss function
        optimizer=trax.optimizers.Adam(0.01), # Optimizer (Don't forget to set LR to 0.01)
        lr_schedule=lr_schedule,
        n_steps_per_checkpoint=10
    )

    eval_task = training.EvalTask(
        labeled_data=eval_gen, # The evaluation generator
        metrics=[ tl.CrossEntropyLoss(),  tl.Accuracy()] # CrossEntropyLoss and Accuracy
    )

    loop = training.Loop(TransformerLM(d_model=d_model,
                                       d_ff=d_ff,
                                       n_layers=n_layers,
                                       n_heads=n_heads,
                                       mode='train'),
                         train_task,
                         eval_tasks=[eval_task],
                         output_dir=output_dir)
    return loop


#
train_batch_stream, eval_batch_stream = load_streams()
# the default used features, trained on 8-bit chars, the length is 8*512=4096 bits maximum of sequence length, to run it on unicode, or different sequence length, you need to pretrain it with required 2*L length, make sure it's power of 2 for best results (see attention is all you need papers for more details)
D_FEATURES=512
CHAR_SIZE=8
MAX_LEN=D_FEATURES*CHAR_SIZE
N_HEADS=8
N_LAYERS=6
D_FF=4*D_FEATURES
DROPOUT=0.1
VOCAB_SIZE=33300
#
parser = ArgumentParser()
parser.add_argument('-t', '--train', type=bool, default=False)
parser.add_argument('-a', '--article', type=str, default=None)
parser.add_argument('-o', '--out', type=str, default='/tmp/out.txt')
parser.add_argument('-so', '--out-summary', type=str, default='/tmp/out_summary.txt')
parser.add_argument('-w', '--wiki', type=str, default=None)
#
args = parser.parse_args()
do_train = args.train
article_file=args.article
out_file=args.out
wiki_title=args.wiki
#
if do_train:
    os.system('rm -f ~/model/model.pkl.gz')
    loop = training_loop(TransformerLM, train_batch_stream, eval_batch_stream)
    loop.run(3000)
model = TransformerLM(mode='eval')
model.init_from_file('model.pkl.gz', weights_only=True)
#

buff=''
summary=''
if article_file is not None:
    with open(article_file, 'r') as a_f:
        buff=a_f.read()
elif wiki_title is not None:
    buff, summary = body_summary(wiki_title)
    print('summary: {}'.format(summary))
out=''
idx=0
MAX=4096
while (idx+1)*MAX <= len(buff):
    passage=buff[idx*MAX:(idx+1)*MAX]
    print(wrapper.fill(passage), '\n')
    O=greedy_decode(passage, model, D_FEATURES)
    print(O)
    out+=O
    idx+=1

with open(out_file, '+w') as o_f:
    o_f.write(out)
