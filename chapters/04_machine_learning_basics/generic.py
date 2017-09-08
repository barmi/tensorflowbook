# TF code scaffolding for building simple models.

import tensorflow as tf

# initialize variables/model parameters

# define the training loop operations
def inference(X):
    # compute inference model over data X and return the result
    return

def loss(X, Y):
    # compute loss over training data X and expected values Y
    loss_val = 0
    return loss_val

def inputs():
    # read/generate input training data X and expected outputs Y
    x_data = [ 1, 2 ]
    y_data = 5
    return x_data, y_data

def train(total_loss):
    # train / adjust model parameters according to computed total loss
    trained_val = 0
    return trained_val

def evaluate(sess, X, Y):
    # evaluate the resulting trained model
    evaluated_val = 0
    return evaluated_val


# Launch the graph in a session, setup boilerplate
with tf.Session() as sess:

    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # actual training loop
    training_steps = 1000
    for step in range(training_steps):
        sess.run([train_op])
        # for debugging and learning purposes, see how the loss gets decremented thru training steps
        if step % 10 == 0:
            print("loss: ", sess.run([total_loss]))

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    sess.close()

