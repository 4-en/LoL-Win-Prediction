import tensorflow as tf

class EnsembleModel(tf.keras.Model):
    def __init__(self, models):
        super(EnsembleModel, self).__init__()
        self.models = models

    def call(self, inputs):
        predictions = [model(inputs) for model in self.models]
        averaged_predictions = tf.reduce_mean(predictions, axis=0)
        return averaged_predictions
