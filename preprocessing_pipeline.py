import string
from string import digits
import unicodedata
import re
import contractions

import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

def text_cleanup(data:list,include_bos_eos_tokens=False,cutoff_index=0, text_max_allowed_len=0):
    _data = []
    exclude = set(string.punctuation)
    currency_symbols = u''.join(chr(i) for i in range(0xffff) if unicodedata.category(chr(i)) == 'Sc')
    for index,sentence in enumerate(data):
        # IDEA: I used this function to change any non-unicode words to ascii-compliant ones using the unicodedata library.
        # IDEA: Here I used the "contractions" library to expand any "you're" etc. to full form. Source: https://github.com/kootenpv/contractions.
        cleaned_sentence = contractions.fix(sentence)
        cleaned_sentence = cleaned_sentence.lower()
        # cleaned_sentence = unicode_to_ascii(cleaned_sentence.lower().strip())
        cleaned_sentence = re.sub("'", '', cleaned_sentence)
        cleaned_sentence = re.sub(",", ' ', cleaned_sentence)
        cleaned_sentence = re.sub("-", ' ', cleaned_sentence)
        cleaned_sentence = re.sub("–", ' ', cleaned_sentence)
        cleaned_sentence = re.sub("\.", ' ', cleaned_sentence)
        cleaned_sentence = re.sub(";", ' ', cleaned_sentence)
        cleaned_sentence = re.sub(" +", ' ', cleaned_sentence)
        cleaned_sentence = re.sub(r"\\", ' ', cleaned_sentence)
        cleaned_sentence = re.sub("/", ' ', cleaned_sentence)

        cleaned_sentence = cleaned_sentence.lstrip()
        cleaned_sentence = cleaned_sentence.rstrip()

        if text_max_allowed_len is not 0:
            splitted_new_sentece = cleaned_sentence.split(' ')
            if len(splitted_new_sentece) > text_max_allowed_len:
                splitted_new_sentece = splitted_new_sentece[:text_max_allowed_len]
                cleaned_sentence = ' '.join(word for word in splitted_new_sentece) 
        
        cleaned_sentence = ''.join(ch for ch in cleaned_sentence if ch not in exclude and currency_symbols)
        remove_digits = str.maketrans('', '', digits)
        cleaned_sentence.translate(remove_digits)

        if include_bos_eos_tokens:
            cleaned_sentence = '<BOS> '+ cleaned_sentence + ' <EOS>'
        _data.append(cleaned_sentence)
        if cutoff_index is not 0 and index >= cutoff_index-1: 
            break
    return _data

def max_length(t):
    return max(len(i) for i in t)


def dataset_preprocessing_pipeline(data:list, include_bos_eos_tokens=False, cutoff_index=0, text_max_allowed_len=0):
    # STEP 1: Clean the text data - remove commas, put all to lowercase etc.
    # ALSO: We provide a cutoff_index if we want to trim the dataset (useful during the development of the model to speed things up and not train on the whole dataset every time)
    # AlSO: we can make the function wrap the original text samples with "<BOS> ... <EOS>" tags.
    newData = text_cleanup(data,include_bos_eos_tokens,cutoff_index,text_max_allowed_len)

    numpy_data = np.asarray(newData)
    tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(corpus_generator= (text for text in newData), target_vocab_size=2**13)

    # def tf_encode(numpy_data):
    #     return tf.py_function(encode, numpy_data, tf.int64)

    tokenized_data = encode(numpy_data,tokenizer)

    # if text_max_allowed_len > 0:
    #     the_longest_texts_length = text_max_allowed_len
    # else:
    the_longest_texts_length = max_length(tokenized_data)
    tokenized_and_padded_text_samples = tf.keras.preprocessing.sequence.pad_sequences(tokenized_data, maxlen=the_longest_texts_length, padding="post", dtype='int64')

    
    return tokenizer, tokenized_and_padded_text_samples, the_longest_texts_length 


def encode(numpy_data, tokenizer):
    tokenized_data = []
    for index,text  in enumerate (numpy_data):
        tokenized_text = [tokenizer.vocab_size] + tokenizer.encode(
            text) + [tokenizer.vocab_size+1]
        tokenized_data.append( tokenized_text)  
    
    # np_tokenized_data = np.asarray(tokenized_data, dtype="int64")
    return tokenized_data