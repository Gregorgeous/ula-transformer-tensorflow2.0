# ======== Global libraries imports ====================
import tensorflow as tf
import tensorflow_datasets as tfds
import time

# # ====== Local code imports for my util functions ======
# from myPickleModule import unpickle #un-comment when debugging this pipeline (to do un-pickling conveniently here)

# =========== Local code imports for text cleanup ======
import string
from string import digits
import unicodedata
import re
import contractions

#  =========== CONSTANTS =======================
EXCLUDE = set(string.punctuation)
CURRENCY_SYMBOLS = u''.join(chr(i) for i in range(0xffff) if unicodedata.category(chr(i)) == 'Sc')
TEXT_MAX_LENGTH = 65
SUMM_MAX_LENGTH = 15
TEXT_TOKENIZER, SUMMARY_TOKENIZER = None, None # Make them empty for now but visible in a global scope.

# =========== PRE-PROCESSING FUNCTIONS =========
def regex(text, isSummary=False):
    """
    Description: 
    Here I perform the text cleanup: convert English contractions (e.g. "you're" to "you are"), get rid of the punctuation etc. 

    Arguments: 
    
    "text": the text sentence - either a summary or an input text. REQUIRED to be in a bytes text format (exampe:_b'this is a text'_)
    
    "isSummary": simple boolean to indicate it's an input text or a summary. This is used only to apply different length if trimming the text in case it's too long. (Bear in mind in a perfect world this should be just an argument specifying the length of the trimming if any, but we can't provide extra arguments in td.dataset.map transformation so this is my workaround)   
    """
    sample = str(text.numpy())
    cleaned_sentence = contractions.fix(sample)
    cleaned_sentence = cleaned_sentence.lower()
    cleaned_sentence = re.sub("'", '', cleaned_sentence)
    cleaned_sentence = re.sub(",", ' ', cleaned_sentence)
    # TODO: consider adding any variations of: " glyph to regex to be changed to a standard : " . https://www.utf8-chartable.de/unicode-utf8-table.pl?start=8192&number=128
    # Currently the regex below wll wipe out every non-standard quotation type as well. 
    cleaned_sentence = re.sub(r"\\xe2\\x80\\x9.", ' ', cleaned_sentence)  
    cleaned_sentence = re.sub("-", ' ', cleaned_sentence)
    cleaned_sentence = re.sub("–", ' ', cleaned_sentence)
    cleaned_sentence = re.sub("\.", ' ', cleaned_sentence)
    cleaned_sentence = re.sub(";", ' ', cleaned_sentence)
    cleaned_sentence = re.sub(" +", ' ', cleaned_sentence)
    cleaned_sentence = re.sub(r"\\", ' ', cleaned_sentence)
    cleaned_sentence = re.sub("/", ' ', cleaned_sentence)

    cleaned_sentence = cleaned_sentence.lstrip()
    cleaned_sentence = cleaned_sentence.rstrip()

    cleaned_sentence = ''.join(ch for ch in cleaned_sentence if ch not in EXCLUDE and CURRENCY_SYMBOLS)
    remove_digits = str.maketrans('', '', digits)
    cleaned_sentence.translate(remove_digits)
    # IDEA: I need to strip the text from the first char. That's because I convert the sentence from bytes format to a string one 
    # (so from b'this a text' to 'this is a text') and somehow it takes that "b"char denoting it's a bytes format as part of the string
    # when doing the conversion "sample = str(text.numpy())" call.    
    cleaned_sentence = cleaned_sentence[1:]
    if isSummary:
        cleaned_and_trimmed_sentence = restrict_length(cleaned_sentence,SUMM_MAX_LENGTH)
    else:
        cleaned_and_trimmed_sentence = restrict_length(cleaned_sentence,TEXT_MAX_LENGTH)

    return cleaned_and_trimmed_sentence.encode()


def restrict_length(cleaned_sentence, text_max_allowed_len):
    if text_max_allowed_len is not 0:
            splitted_new_sentece = cleaned_sentence.split(' ')
            if len(splitted_new_sentece) > text_max_allowed_len:
                splitted_new_sentece = splitted_new_sentece[:text_max_allowed_len]
                trimmed_cleaned_sentence = ' '.join(word for word in splitted_new_sentece) 
                return trimmed_cleaned_sentence
    return cleaned_sentence

def max_length_summaries(t):
    return max(len(summaries) for texts,summaries in t)

def max_length_texts(t):
    return max(len(texts) for texts,summaries in t)

def filter_max_length(text, summary, text_max_length=TEXT_MAX_LENGTH, summ_max_length = SUMM_MAX_LENGTH ):
  return tf.logical_and(tf.size(text) <= text_max_length,
                        tf.size(summary) <= summ_max_length)

# =============== DATASET PIPELINE ==================================================
def dataset_preprocessing_pipeline(texts:list,summaries:list, cutoff_index=0, texts_max_length = 65, summaries_max_length= 15,  batch_size=64, buffer_size=20000):
    # ------------ Re-initialise some global variables -----------------------
    # (yes, this is not the "cleanest" approach but we can't add extra arguments
    # to the dataset's tf.data transformations and therefore need to rely on those global-scope variables for any extra logic like the "text_max_allowed" ...)
    global TEXT_MAX_LENGTH, SUMM_MAX_LENGTH, TEXT_TOKENIZER, SUMMARY_TOKENIZER
    TEXT_MAX_LENGTH = texts_max_length
    SUMM_MAX_LENGTH = summaries_max_length

    # ------------ Transform the Python Lists to Tf.dataset ------------------ 
    dataset = tf.data.Dataset.from_tensor_slices((texts,summaries))
    if cutoff_index is not 0 and cutoff_index < len(texts):
        # If cutoff_index specified, take only as many samples as specified 
        dataset = dataset.take(cutoff_index)

    # ------------ Specify extra functions that for which the newly created text/summary tokenisers NEED to be in the scope. ------ 
    def BPE_encoding(lang1, lang2):
        lang1 = [TEXT_TOKENIZER.vocab_size] + TEXT_TOKENIZER.encode(
            lang1.numpy()) + [TEXT_TOKENIZER.vocab_size+1]

        lang2 = [SUMMARY_TOKENIZER.vocab_size] + SUMMARY_TOKENIZER.encode(
            lang2.numpy()) + [SUMMARY_TOKENIZER.vocab_size+1]
        
        return lang1, lang2

    def text_and_summary_cleanup(text, summary):
        cleaned_and_trimmed_text = regex(text, False)
        cleaned_and_trimmed_summary = regex(summary, True)
        return cleaned_and_trimmed_text, cleaned_and_trimmed_summary
    
    # IDEA: Since tf.data "map()"" operates in graph mode, we need to wrap it in "py_function" where we can feeely execute 
    # any python code - in our case that's necessary as we want to do RegEx queries and clean the text. 
    # IDEA: This "execute python code in TF" wrapper is for the text_and_summary_cleanup method
    def tfds_map_py_wrapper(text, summary):
        return tf.py_function(text_and_summary_cleanup, [text, summary], [tf.string, tf.string])
    # IDEA: this wrapper is for performing the BPE tokenisation.
    def tfds_map_py_wrapper2(text, summary):
        return tf.py_function(BPE_encoding, [text, summary], [tf.int64, tf.int64])
    # --------------- DATASET TRANSFORMATIONS PIPELINE ------------------
    # Step 1: Clean the text using text_and_summary_cleanup method wrapped in the TensorFlow's utility tfds_map_py_wrapper.  
    dataset = dataset.map(tfds_map_py_wrapper)
    # Step 2: Initialise the BPE tokenisers on texts and summary only now, so it builds its vocabulary on the text data that 
    # was already pre-processed (otherwise we would end up having word-as-token mappings of words that we won't need anyway) 
    TEXT_TOKENIZER = tfds.features.text.SubwordTextEncoder.build_from_corpus((text.numpy() for text, summary in dataset), target_vocab_size=2**13)
    SUMMARY_TOKENIZER = tfds.features.text.SubwordTextEncoder.build_from_corpus((summary.numpy() for text, summary in dataset), target_vocab_size=2**13)
    dataset = dataset.map(tfds_map_py_wrapper2)
    # Step 3: Establish the longest text length in the dataset's samples to then correctly align the whole dataset at the padding step in "padded_batch" function
    # (NOTE: Yes, theoretically this step should be redundant as you can provide padded_shapes 
    # argument in the padded_batch() transformation with "[-1]" and it should perform the exactly same logic .. but apparently it didn't for the summaries when I inspected the output. So here I ensure the max length and specify such in the padded_shape in the next step)
    texts_bpe_encodings_max_length = max_length_texts(dataset)
    summaries_bpe_encodings_max_length = max_length_summaries(dataset)
    # Step 4: Shuffle the dataset, pad all the text and summary data samples to the same length (each has it's own appropriate one provided earlier), and form it all into batches.
    dataset = dataset.padded_batch(
    batch_size, padded_shapes=([texts_bpe_encodings_max_length], [summaries_bpe_encodings_max_length]))
    # Step 5: return all the objects we'll later need in out MAIN file. 
    return dataset,TEXT_TOKENIZER, SUMMARY_TOKENIZER,texts_bpe_encodings_max_length,summaries_bpe_encodings_max_length
