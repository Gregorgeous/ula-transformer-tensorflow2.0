import tensorflow as tf
K = tf.keras.backend # Alias to Keras' backend namespace.

class PerplexityMetric(tf.keras.metrics.Metric):
    """
    USAGE NOTICE: this metric accepts only logits for now (i.e. same way that in tf.keras.losses.SparseCategoricalCrossentropy from_logits is 'False', here it's enforced to be always True so you need to provide it in such a format)
    METRIC DESCRIPTION:
    Popular metric for evaluating language modelling architectures.
    More info: http://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf.
    DISCLAIMER: Original function created by Kirill Mavreshko in https://github.com/kpot/keras-transformer/blob/b9d4e76c535c0c62cadc73e37416e4dc18b635ca/example/run_gpt.py#L106. 
    My "contribution": I converted Kirill method's logic into a Tensorflow 2.0 way of doing things - this requireD making the metric a fully-fledged object by subclassing the Metric object. 
    """
    def __init__(self, name='perplexity', **kwargs):
      super(PerplexityMetric, self).__init__(name=name, **kwargs)
      self.cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
      self.perplexity = self.add_weight(name='tp', initializer='zeros')

    # @tf.function
    def _calculate_perplexity(self, real, pred):
      mask = tf.math.logical_not(tf.math.equal(real, 0))
      loss_ = self.cross_entropy(real, pred)
      mask = tf.cast(mask, dtype=loss_.dtype)
      loss_ *= mask
      step1 = K.mean(loss_, axis=-1)
      step2 = K.exp(step1)
      perplexity = K.mean(step2)

      return perplexity 


    def update_state(self, y_true, y_pred, sample_weight=None):
      # TODO:FIXME: handle sample_weight ! 
      if sample_weight is not None:
          print("WARNING! Provided 'sample_weight' argument to the perplexity metric. Currently this is not handled and won't do anything differently..")
      perplexity = self._calculate_perplexity(y_true, y_pred)
      self.perplexity.assign_add(perplexity)
        
    def result(self):
      return self.perplexity

    def reset_states(self):
      # The state of the metric will be reset at the start of each epoch.
      self.perplexity.assign(0.)
