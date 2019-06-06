# ======== Global libraries imports ====
import tensorflow as tf
import tensorflow_datasets as tfds
import time
import numpy as np
import matplotlib.pyplot as plt

# ====== Local code imports for Transformer ============
from custom_loss import loss_object,loss_function,train_loss,train_accuracy
from dynamic_learning_rate import CustomSchedule
from transformer import Transformer
from create_all_masks import create_masks
from plot_attention import plot_attention_weights
# ====== Local import from my preprocessing pipeline ===
# (which is half my code, half one from T2.0 transformer tutorial )
from preprocessing_pipeline import dataset_preprocessing_pipeline

# ====== Local code imports for my util functions ======
from myPickleModule import unpickle

#  ============= OUR DATASET LOADUP STEP ===================
print("===== START OF THE DATASET SAMPLES PREPROCESSING =========== ")
target_summaries = unpickle('allPreprocessedSummaries')
input_texts = unpickle('allPreprocessedTexts')
start = time.time()
print("BEGUN: preprocessing pipeline on inputs")
text_tokenizer, all_tokenized_texts, texts_max_length = dataset_preprocessing_pipeline(input_texts,include_bos_eos_tokens=False, cutoff_index=20, text_max_allowed_len=65)
print(f"FINISHED: Time taken: {(time.time() - start):.3}s ")
print('-----------------')
print("BEGUN: preprocessing pipeline on summaries")
summaries_tokenizer, all_tokenized_summaries, summaries_max_length = dataset_preprocessing_pipeline(target_summaries,include_bos_eos_tokens=False, cutoff_index=20, text_max_allowed_len=10)
print(f"FINISHED: Time taken: {(time.time() - start):.3}s ")
print('-----------------')


# ================= MANUALLY RESHAPE THE DATA INTO BATCHES ================
# (this step is needed because I decided to feed the model with the in-memory numpy array feeding instead of the "tf.data"-based approach TF team shown in the Transformer tutorial)
BATCH_SIZE = 4
all_tokenized_texts = np.reshape(all_tokenized_texts,(BATCH_SIZE,-1,texts_max_length))
all_tokenized_summaries = np.reshape(all_tokenized_summaries, (BATCH_SIZE,-1,summaries_max_length))

# ============== MODEL HYPERPARAMETERS =====================
num_layers = 4
d_model = 128
dff = 512
num_heads = 8

# To un-comment once train data properly BPE encoded
input_vocab_size = text_tokenizer.vocab_size + 2
target_vocab_size = summaries_tokenizer.vocab_size + 2
# FOR TESTING ONLY 
# input_vocab_size = 100
# target_vocab_size = 100
dropout_rate = 0.1
# ==========================================================

learning_rate = CustomSchedule(d_model)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, 
                                     epsilon=1e-9)

# ============ INITIALISE TRANSFORMER MODEL ============
transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size, dropout_rate)
# ============ CHECKPOINTING ==================
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')

# ================= TRAINING TIME ! ===================
EPOCHS = 20

@tf.function
def train_step(inp, tar):
  tar_inp = tar[:, :-1]
  tar_real = tar[:, 1:]
  
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  
  with tf.GradientTape() as tape:
    predictions, _ = transformer(inp, tar_inp, 
                                 True, 
                                 enc_padding_mask, 
                                 combined_mask, 
                                 dec_padding_mask)
    loss = loss_function(tar_real, predictions)

  gradients = tape.gradient(loss, transformer.trainable_variables)    
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
  
  train_loss(loss)
  train_accuracy(tar_real, predictions)

for epoch in range(EPOCHS):
  start = time.time()
  
  train_loss.reset_states()
  train_accuracy.reset_states()
  
  # inp -> portuguese, tar -> english
  for (batch, (inp, tar)) in enumerate(zip(all_tokenized_texts,all_tokenized_summaries)):
    train_step(inp, tar)
    
    if batch % 500 == 0:
      print ('Epoch {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
          epoch + 1, batch, train_loss.result(), train_accuracy.result()))
      
  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))
    
  print ('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, 
                                                train_loss.result(), 
                                                train_accuracy.result()))

  print ('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))


# ============ INFERENCE TIME ! Predict summaries ===================

def evaluate(inp_sentence):
  start_token = [text_tokenizer.vocab_size]
  end_token = [text_tokenizer.vocab_size + 1]
  
  # inp sentence is portuguese, hence adding the start and end token
  inp_sentence = start_token + text_tokenizer.encode(inp_sentence) + end_token
  encoder_input = tf.expand_dims(inp_sentence, 0)
  
  # as the target is english, the first word to the transformer should be the
  # english start token.
  decoder_input = [summaries_tokenizer.vocab_size]
  output = tf.expand_dims(decoder_input, 0)
    
  for i in range(summaries_max_length):
    # TODO: beware! The TF2.0 Transformer tutorial processes the input data so that both the Portugal input (in mine:text) and the English output (in mine: summary) are of the same length. I prefer having two separate lengths so this loop is relies on the summaries' max possible length. HOWEVER, the below masking process might mess things up potentially (as it assumes equal length of both).  
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)
  
    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights = transformer(encoder_input, 
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)
    
    # select the last word from the seq_len dimension
    predictions = predictions[: ,-1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.cast(tf.argmax(predictions, axis=-1), tf.int32)
    
    # return the result if the predicted_id is equal to the end token
    if tf.equal(predicted_id, summaries_tokenizer.vocab_size+1):
      return tf.squeeze(output, axis=0), attention_weights
    
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

  return tf.squeeze(output, axis=0), attention_weights


def translate(sentence, plot=''):
    result, attention_weights = evaluate(sentence)

    predicted_sentence = summaries_tokenizer.decode([i for i in result 
                                                if i < summaries_tokenizer.vocab_size])  

    print('Input: {}'.format(sentence))
    print('Predicted translation: {}'.format(predicted_sentence))

    if plot:
        plot_attention_weights(attention_weights, sentence, result, plot)

    