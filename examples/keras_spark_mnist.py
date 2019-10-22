from __future__ import print_function

import argparse
import os
import subprocess

from pyspark import SparkConf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import OneHotEncoderEstimator
from pyspark.sql import SparkSession

from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D

import horovod.spark.keras as hvd
from horovod.spark.common.store import LocalStore, HDFSStore

parser = argparse.ArgumentParser(description='Keras Spark MNIST Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--master',
                    help='spark master to connect to')
parser.add_argument('--num-proc', type=int,
                    help='number of worker processes for training, default: `spark.default.parallelism`')
parser.add_argument('--batch-size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=12,
                    help='number of epochs to train')
parser.add_argument('--work-dir', default='/tmp',
                    help='temporary working directory to write intermediate files (prefix with hdfs:// to use HDFS)')

args = parser.parse_args()

conf = SparkConf().setAppName('keras_spark_mnist').set('spark.sql.shuffle.partitions', '16')
if args.master:
    conf.setMaster(args.master)
spark = SparkSession.builder.config(conf=conf).getOrCreate()

work_dir = args.work_dir
hdfs_prefix = 'hdfs://'
if work_dir.startswith(hdfs_prefix):
    work_dir = work_dir[len(hdfs_prefix):]
    store = HDFSStore(work_dir)
else:
    store = LocalStore(work_dir)

data_url = 'https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.bz2'
libsvm_path = os.path.join(work_dir, 'mnist.bz2')
if not os.path.exists(libsvm_path):
    subprocess.check_output(['wget', data_url, '-O', libsvm_path])

df = spark.read.format('libsvm') \
    .option('numFeatures', '784') \
    .load(libsvm_path)

encoder = OneHotEncoderEstimator(inputCols=['label'],
                                 outputCols=['label_vec'],
                                 dropLast=False)
model = encoder.fit(df)
train_df = model.transform(df)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

optimizer = keras.optimizers.Adadelta(1.0)
loss = keras.losses.categorical_crossentropy

store = LocalStore('/tmp')
keras_estimator = hvd.KerasEstimator(num_proc=args.num_proc,
                                     store=store,
                                     model=model,
                                     optimizer=optimizer,
                                     loss=loss,
                                     metrics=['accuracy'],
                                     feature_cols=['features'],
                                     label_cols=['label_vec'],
                                     batch_size=128,
                                     epochs=12,
                                     verbose=1)

keras_model = keras_estimator.fit(train_df).setOutputCols(['y_pred'])

# pred_df = keras_model.transform(test_df)
# evaluator = BinaryClassificationEvaluator(rawPredictionCol='y_pred', labelCol='y')
# print('Test area under ROC:', evaluator.evaluate(pred_df))

spark.stop()
