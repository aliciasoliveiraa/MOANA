import tensorflow as tf

class TFFedProxLoss(tf.keras.losses.Loss):
    def __init__(self, mu=0.01, name="fedprox_loss"):
        """Compute FedProx loss: a loss penalizing the deviation from global model.

        Args:
            mu: weighting parameter
            name: name of the loss
        """
        super().__init__(name=name)
        if mu < 0.0:
            raise ValueError("mu should be no less than 0.0")
        self.mu = mu

    def call(self, input_model, target_model):
        """Forward pass in training.

        Args:
            input_model (tf.keras.Model): the local tensorflow model
            target_model (tf.keras.Model): the copy of global tensorflow model when local clients received it
                                           at the beginning of each local round

        Returns:
            FedProx loss term
        """
        prox_loss = 0.0        
        for input_param, target_param in zip(input_model.variables, target_model.variables):
            prox_loss += (self.mu / 2) * tf.reduce_sum(tf.square(input_param - target_param))
                     
            '''
            squared_difference = tf.square(input_param - target_param)
            print(f'input_param: {input_param.numpy()}, target_param: {target_param.numpy()}')
            print(f'squared_difference: {squared_difference.numpy()}')
            
            prox_loss += (self.mu / 2) * tf.reduce_sum(squared_difference)
            print(f'prox_loss: {prox_loss.numpy()}')
            '''
        
        return prox_loss

