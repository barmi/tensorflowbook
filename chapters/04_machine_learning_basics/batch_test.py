import tensorflow as tf
import numpy as np
import sys

TRAINING_FILE = 'flight.csv'


## read training data and label
def read_data(file_name):
    try:
        csv_file = tf.train.string_input_producer([file_name], name='filename_queue')
        textReader = tf.TextLineReader()
        _, line = textReader.read(csv_file)
        year, flight, time = tf.decode_csv(line, record_defaults=[[1900], [""], [0]], field_delim=',')
    except:
        print("Unexpected error:", sys.exc_info()[0])
        exit()
    return year, flight, time


def read_data_batch(file_name, batch_size=10):
    year, flight, time = read_data(file_name)
    batch_year, batch_flight, batch_time = tf.train.batch([year, flight, time], batch_size=batch_size)

    return batch_year, batch_flight, batch_time


def main():
    print('start session')
    # coornator 위에 코드가 있어야 한다
    # 데이타를 집어 넣기 전에 미리 그래프가 만들어져 있어야 함.
    batch_year, batch_flight, batch_time = read_data_batch(TRAINING_FILE)
    year = tf.placeholder(tf.int32, [None, ])
    flight = tf.placeholder(tf.string, [None, ])
    time = tf.placeholder(tf.int32, [None, ])

    tt = time * 10

    with tf.Session() as sess:
        try:

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            for i in range(10):
                y_, f_, t_ = sess.run([batch_year, batch_flight, batch_time])
                print(sess.run(tt, feed_dict={time: t_}))

            print('stop batch')
            coord.request_stop()
            coord.join(threads)
        except:
            print("Unexpected error:", sys.exc_info()[0])


main()
