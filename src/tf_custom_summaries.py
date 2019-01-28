import tensorflow as tf
from tensorboard import summary as summary_lib
from tensorboard.plugins.custom_scalar import layout_pb2

def class_score_mmm(name='class_score'):
    # Create a custom scalar summary as a margin chart.
    # Displays the mean value and the min and max values as the upper and lower limits.

    tf_class_scores = tf.placeholder(tf.float32, shape=(None,))
    tf_mean_op = summary_lib.scalar(name, tf.math.reduce_mean(tf_class_scores))
    tf_min_op = summary_lib.scalar(name + '/min', tf.math.reduce_min(tf_class_scores))
    tf_max_op = summary_lib.scalar(name + '/max', tf.math.reduce_max(tf_class_scores))

    tf_class_score_chart = layout_pb2.Chart(
                                title=name,
                                margin=layout_pb2.MarginChartContent(
                                    series=[
                                        layout_pb2.MarginChartContent.Series(
                                            value=tf_mean_op.name.split(':')[0], #name + '/scalar_summary',
                                            lower=tf_min_op.name.split(':')[0],  #name + '/min/scalar_summary',
                                            upper=tf_max_op.name.split(':')[0]   #name + '/max/scalar_summary'
                                        ),
                                    ],))

    return tf_class_scores, tf_class_score_chart