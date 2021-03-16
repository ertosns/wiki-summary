import sys
import os
import numpy as np
import textwrap
wrapper = textwrap.TextWrapper(width=70)
import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp
from trax.supervised import training
import wikipedia as wiki

os.environ['NO_GCE_CHECK'] = 'true'



# Special tokens
SEP = 0 # Padding or separator token
EOS = 1 # End of sentence token


def tokenize(input_str, EOS=1):
    """Input str to features dict, ready for inference"""

    # It takes streams and returns streams,
    # we get around it by making a 1-element stream with `iter`.
    inputs =  next(trax.data.tokenize(iter([input_str]),
                                      vocab_dir='vocab_dir/',
                                      vocab_file='summarize32k.subword.subwords'))

    # Mark the end of the sentence with EOS
    return list(inputs) + [EOS]

def detokenize(integers):
    """List of ints to str"""

    s = trax.data.detokenize(integers,
                             vocab_dir='vocab_dir/',
                             vocab_file='summarize32k.subword.subwords')

    return wrapper.fill(s)

# Concatenate tokenized inputs and targets using 0 as separator.
def preprocess(stream):
    for (article, summary) in stream:
        joint = np.array(list(article) + [EOS, SEP] + list(summary) + [EOS])
        mask = [0] * (len(list(article)) + 2) + [1] * (len(list(summary)) + 1) # Accounting for EOS and SEP
        yield joint, joint, np.array(mask)

def create_tensor(t):
    """Create tensor from list of lists"""
    return jnp.array(t)


def display_tensor(t, name):
    """Display shape and tensor"""
    print(f'{name} shape: {t.shape}\n')
    print(f'{t}\n')


def load_streams():
    # Importing CNN/DailyMail articles dataset
    train_stream_fn = trax.data.TFDS('cnn_dailymail',
                                     data_dir='data/',
                                     keys=('article', 'highlights'),
                                     train=True)
    eval_stream_fn = trax.data.TFDS('cnn_dailymail',
                                    data_dir='data/',
                                    keys=('article', 'highlights'),
                                    train=False)


    # You can combine a few data preprocessing steps into a pipeline like this.
    # @see https://trax-ml.readthedocs.io/en/latest/trax.data.html
    input_pipeline = trax.data.Serial(
        # Tokenizes
        trax.data.Tokenize(vocab_dir='vocab_dir/',
                           vocab_file='summarize32k.subword.subwords'),
        preprocess,
        # Filters out examples longer than 2048
        trax.data.FilterByLength(2048)
    )

    # Apply preprocessing to data streams.
    train_stream = input_pipeline(train_stream_fn())
    eval_stream = input_pipeline(eval_stream_fn())

    train_input, train_target, train_mask = next(train_stream)

    # bucket batches into batches of similar boundaries length.
    boundaries =  [128, 256,  512, 1024]
    batch_sizes = [16,    8,    4,    2, 1]
    # Create the streams.
    train_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes)(train_stream)
    eval_batch_stream = trax.data.BucketByLength(
        boundaries, batch_sizes)(eval_stream)
    return train_batch_stream, eval_batch_stream

def next_symbol(cur_output_tokens, model, d_model):
    """Returns the next symbol for a given sentence.

    Args:
        cur_output_tokens (list): tokenized sentence with EOS and PAD tokens at the end.
        model (trax.layers.combinators.Serial): The transformer model.

    Returns:
        int: tokenized symbol.
    """

    token_length = len(cur_output_tokens)
    padded_length = 2**int(np.ceil(np.log2(token_length + 1)))
    assert(padded_length<=d_model*8, 'the d_features is 512 char, or 4096 bit maximum') #assuming 8bit char
    #print('token length: {}, padded length: {}'.format(token_length, padded_length))
    padded = cur_output_tokens + [0] * (padded_length - token_length)
    padded_with_batch = np.array(padded)[None, :] # Don't replace this 'None'! This is a way of setting the batch dim

    output, _ = model((jnp.array(padded_with_batch), jnp.array(padded_with_batch)))
    log_probs = output[0, len(cur_output_tokens), :]

    return int(np.argmax(log_probs))


def greedy_decode(input_sentence, model, d_model):
    """Greedy decode function.

    Args:
        input_sentence (string): a sentence or article.
        model (trax.layers.combinators.Serial): Transformer model.

    Returns:
        string: summary of the input.
    """
    cur_output_tokens = tokenize(input_sentence) + [0]
    generated_output = []
    cur_output = 0
    EOS = 1

    while cur_output != EOS:
        cur_output = next_symbol(cur_output_tokens, model, d_model)
        cur_output_tokens.append(cur_output)
        generated_output.append(cur_output)
        print(detokenize(generated_output))

    return detokenize(generated_output)

def body_summary(title):
    try:
        suggest = wiki.search(title, results=1)
        page=wiki.page(suggest[0])
    except:
        print('no page with given title: {}, here are some suggestions: [{}]'.format(title, suggest))
    return page.content, page.summary
