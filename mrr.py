# TensorFlow 2 implementation of Mean Reciprocal Rank(MRR) metric.

import tensorflow as tf


def mrr_fn(y_true, y_pred):
    # Function to calculate MRR.
    results_sorted = tf.math.top_k(y_pred, k=tf.shape(y_pred)[1], sorted=True)[1]
    positions = tf.where(tf.equal(results_sorted, tf.cast(y_true, dtype=tf.int32)))[:, 1]
    ranks = 1 / (positions + 1)
    return ranks


class MeanReciprocalRank(tf.keras.metrics.Metric):

    def __init__(self, name='mrr', **kwargs):
        super(MeanReciprocalRank, self).__init__(name=name, **kwargs)
        self.mrr_fn = mrr_fn
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.mrr_fn(y_true, y_pred)
        self.total.assign_add(tf.cast(tf.reduce_sum(metric), dtype=tf.float32))
        self.count.assign_add(tf.cast(tf.size(y_true), dtype=tf.float32))

    def result(self):
      return self.total / self.count
