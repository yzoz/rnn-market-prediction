import pprint
import tensorflow as tf
import tensorflow.contrib.slim as slim

from rnn import RNN

"""
ABC = '2017-11-06T00:00:00'
XYZ = '2017-11-16T12:00:00'

ABC = '2017-11-06T00:00:00'
XYZ = '2017-11-27T00:00:00'
"""

lr_start = 1.0
lr_need = 0.00001
epoch = 1000

lr_decay = round((lr_need/lr_start)**(1/epoch), 5)

flags = tf.app.flags

flags.DEFINE_string('model', 'rts-60x7', 'Model Name')
flags.DEFINE_string('abc', '2015-01-01T00:00:00', 'Start')
flags.DEFINE_string('xyz', '2017-12-15T00:00:00', 'Finish')

flags.DEFINE_float('starter_learning_rate', lr_start, 'Initial learning rate')
flags.DEFINE_float('decay', lr_decay, 'Learning rate exponencial decay')
flags.DEFINE_integer('iters', epoch, 'Number of iterations (epochs)')
flags.DEFINE_integer('display_step', 1, 'Display Info every [n] steps')

flags.DEFINE_integer('batch', 100, 'Mini batch size')
flags.DEFINE_string('win', '60m', 'Candles Timeframe')
flags.DEFINE_integer('future', 1, 'Prediction interval')
flags.DEFINE_integer('backward', 7, 'Back to the past')

flags.DEFINE_integer('n_hidden', 512, 'RNN size')
flags.DEFINE_integer('n_layers', 2, 'Number or Cells')
flags.DEFINE_integer('n_states', 3, 'How many states can be')

flags.DEFINE_string('wn', 'train', 'train :: cont :: test :: write :: pred')

flags.DEFINE_boolean('writer', False, 'Write or not Summarys')

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def show_all_variables():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def main(_):
    pp.pprint(flags.FLAGS.__flags)
    graph = tf.Graph()
    with graph.as_default():
        with tf.Session(graph=graph) as sess:
            net = RNN(
            sess,
            FLAGS.model,
            FLAGS.abc,
            FLAGS.xyz,
            FLAGS.starter_learning_rate,
            FLAGS.decay,
            FLAGS.iters,
            FLAGS.display_step,
            FLAGS.batch,
            FLAGS.win,
            FLAGS.future,
            FLAGS.backward,
            FLAGS.n_hidden,
            FLAGS.n_layers,
            FLAGS.n_states,
            FLAGS.wn,
            FLAGS.writer
        )
            show_all_variables()
            if FLAGS.wn == 'train' or FLAGS.wn == 'cont':
                net.run()
            else:
                net.test()

if __name__ == "__main__":
    tf.app.run()