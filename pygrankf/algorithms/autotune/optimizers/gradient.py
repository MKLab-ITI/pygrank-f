from pygrankf.core import backend, utils


def tfsgd(loss,
          max_vals,
          min_vals,
          starting_parameters,
          epochs: int = 5000,
          patience: int = 100,
          tfoptimizer=None,
          **kwargs):
    import tensorflow as tf
    optimizer = tf.optimizers.Adam(0.01) if tfoptimizer is None else tfoptimizer
    with backend.Backend("tensorflow"):
        parameters = [tf.Variable(value, dtype=tf.float32) for value in starting_parameters]
        best_params = parameters
        best_loss = float('inf')
        curr_patience = patience
        for epoch in range(epochs):
            with tf.GradientTape() as tape:
                epoch_loss = loss(parameters)
            if best_loss > epoch_loss:
                utils.log(f"Epoch {epoch} loss {float(epoch_loss)}")
                best_loss = epoch_loss
                curr_patience = patience
                best_params = [float(value.numpy()) for value in parameters]
            gradients = tape.gradient(epoch_loss, parameters)
            optimizer.apply_gradients(zip(gradients, parameters))
            for var, mm, mx in zip(parameters, min_vals, max_vals):
                if var < mm:
                    var.assign(mm)
                if var > mx:
                    var.assign(mx)
            curr_patience -= 1
            if curr_patience == 0:
                break
    utils.log()
    return best_params
