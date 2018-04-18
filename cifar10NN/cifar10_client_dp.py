"""A binary to train CIFAR-10 with DP.
"""
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function
import os
from datetime import datetime
import time
import pickle
import sys
import tensorflow as tf
import numpy as np
import json

import cifar10NN

from cifar10NN import cifar10, cifar10_eval, cifar10_input

from dp.differential_privacy.dp_sgd.dp_optimizer import dp_optimizer
from dp.differential_privacy.dp_sgd.dp_optimizer import sanitizer
from dp.differential_privacy.dp_sgd.dp_optimizer import utils
from dp.differential_privacy.privacy_accountant.tf import accountant

FLAGS = tf.flags.FLAGS
IMAGE_SIZE = 24

"""BEGIN PARSER FLAGS"""
weights = []
central_weights = []
client = -1
network_model = None
"""END PARSER FLAGS"""

"""BEGIN TF FLAGS"""
# parameters for the training
tf.flags.DEFINE_integer("batch_size", 1000,
                        "The training batch size.")
tf.flags.DEFINE_integer("batches_per_lot", 1,
                        "Number of batches per lot.")
# Together, batch_size and batches_per_lot determine lot_size.
tf.flags.DEFINE_integer("num_training_steps", 100,
                        "The number of training steps."
                        "This counts number of lots.")

tf.flags.DEFINE_bool("randomize", True,
                     "If true, randomize the input data; otherwise use a fixed "
                     "seed and non-randomized input.")
tf.flags.DEFINE_bool("freeze_bottom_layers", False,
                     "If true, only train on the logit layer.")
tf.flags.DEFINE_bool("save_mistakes", False,
                     "If true, save the mistakes made during testing.")
tf.flags.DEFINE_float("lr", 0.05, "start learning rate")
tf.flags.DEFINE_float("end_lr", 0.05, "end learning rate")
tf.flags.DEFINE_float("lr_saturate_epochs", 0,
                      "learning rate saturate epochs; set to 0 for a constant "
                      "learning rate of --lr.")

# For searching parameters
tf.flags.DEFINE_integer("num_hidden_layers", 2,
                        "Number of hidden layers in the network")
tf.flags.DEFINE_integer("hidden_layer_num_units", 384,
                        "Number of units per hidden layer")
tf.flags.DEFINE_float("default_gradient_l2norm_bound", 3.0, "norm clipping")

tf.flags.DEFINE_string("training_data_path",
                       "data/cifar_data/cifar10_train.tfrecord",
                       "Location of the training data.")
tf.flags.DEFINE_string("eval_data_path",
                       "data/cifar_data/cifar10_test.tfrecord",
                       "Location of the eval data.")
tf.flags.DEFINE_integer("eval_steps", 100,
                        "Evaluate the model every eval_steps")

# Parameters for privacy spending. We allow linearly varying eps during
# training.
tf.flags.DEFINE_string("accountant_type", "Moments", "Moments, Amortized.")

# Flags that control privacy spending during training.
tf.flags.DEFINE_float("eps", 1.0,
                      "Start privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("end_eps", 1.0,
                      "End privacy spending for one epoch of training, "
                      "used if accountant_type is Amortized.")
tf.flags.DEFINE_float("eps_saturate_epochs", 0,
                      "Stop varying epsilon after eps_saturate_epochs. Set to "
                      "0 for constant eps of --eps. "
                      "Used if accountant_type is Amortized.")
tf.flags.DEFINE_float("delta", 1e-5,
                      "Privacy spending for training. Constant through "
                      "training, used if accountant_type is Amortized.")
tf.flags.DEFINE_float("sigma", 6.0,
                      "Noise sigma, used only if accountant_type is Moments")

tf.flags.DEFINE_string("target_eps", "2,4",
                       "Log the privacy loss for the target epsilon's. Only "
                       "used when accountant_type is Moments.")
tf.flags.DEFINE_float("target_delta", 1e-5,
                      "Maximum delta for --terminate_based_on_privacy.")
tf.flags.DEFINE_bool("terminate_based_on_privacy", True,
                     "Stop training if privacy spent exceeds "
                     "(max(--target_eps), --target_delta), even "
                     "if --num_training_steps have not yet been completed.")

tf.flags.DEFINE_string("save_path", "models/cifar/clients",
                       "Directory for saving model outputs.")
tf.flags.DEFINE_string("load_path", "models/central",
                       "Directory for loading model outputs.")

TFFLAGS = tf.flags.FLAGS
"""END TF FLAGS"""



class cifar10_client(object):
    
    def __init__(self, index, num_training_images):#, num_testing_images):
        self.index = index
        self.client = index
        self.num_training_images = num_training_images
        self.num_testing_images = num_training_images#num_testing_images
        if network_model==None: buildNetwork()
        self.results = []


    def train(self,  network_parameters=network_model, eval_steps=100):
        """Train CIFAR-10 for a number of steps.

        Args:
        cifar_train_file: path of CIFAR-10 train data file.
        cifar_test_file: path of CIFAR-10 test data file.
        network_parameters: parameters for defining and training the network.
        save_path: path where to save trained parameters.
        eval_steps: evaluate the model every eval_steps.

        Returns:
        the result after the final training step.

        Raises:
        ValueError: if the accountant_type is not supported.
        """
        #model_dir=TFFLAGS.train_dir
        #self.model_dir=model_dir
        network_parameters=buildNetwork()
        model_path='models/central/'
        save_path="models/cifar/clients"
        cifar_train_file="data/clients/cifar"
        cifar_test_file="data/clients/cifar"
        
        batch_size = TFFLAGS.batch_size

        params = {"accountant_type": TFFLAGS.accountant_type,
            "task_id": 0,
            "batch_size": TFFLAGS.batch_size,
            "default_gradient_l2norm_bound":
            network_parameters.default_gradient_l2norm_bound,
            "num_hidden_layers": TFFLAGS.num_hidden_layers,
            "hidden_layer_num_units": TFFLAGS.hidden_layer_num_units,
            "num_examples": self.num_training_images,
            "learning_rate": TFFLAGS.lr,
            "end_learning_rate": TFFLAGS.end_lr,
            "learning_rate_saturate_epochs": TFFLAGS.lr_saturate_epochs
           }

        params.update({"sigma": TFFLAGS.sigma})
        #saver = tf.train.Saver()

        with tf.Graph().as_default(), tf.Session() as sess, tf.device('/cpu:0'):
            # Create the basic Cifar model.
            # TODO: GET INPUT FOR CLIENT
            #images, labels = CifarInput(cifar_train_file, batch_size, TFFLAGS.randomize)
            images, labels = cifar10_input.distorted_inputs("data/clients/cifar", self.index, TFFLAGS.batch_size)
            

            logits, projection, training_params = utils.BuildNetwork(
                images, network_parameters)
            
            
            cost = tf.nn.softmax_cross_entropy_with_logits(
                logits=logits, labels=tf.one_hot(labels, 10))

            # The actual cost is the average across the examples.
            cost = tf.reduce_sum(cost, [0]) / batch_size

            priv_accountant = accountant.GaussianMomentsAccountant(self.num_training_images)
            sigma = TFFLAGS.sigma
            with_privacy = TFFLAGS.sigma > 0
            with_privacy = False
            # Note: Here and below, we scale down the l2norm_bound by
            # batch_size. This is because per_example_gradients computes the
            # gradient of the minibatch loss with respect to each individual
            # example, and the minibatch loss (for our model) is the *average*
            # loss over examples in the minibatch. Hence, the scale of the
            # per-example gradients goes like 1 / batch_size.
            gaussian_sanitizer = sanitizer.AmortizedGaussianSanitizer(
                priv_accountant, 
                [network_parameters.default_gradient_l2norm_bound / batch_size, True])

            for var in training_params:
                if "gradient_l2norm_bound" in training_params[var]:
                    l2bound = training_params[var]["gradient_l2norm_bound"] / batch_size
                    gaussian_sanitizer.set_option(var,
                                                  sanitizer.ClipOption(l2bound, True))
            lr = tf.placeholder(tf.float32)
            eps = tf.placeholder(tf.float32)
            delta = tf.placeholder(tf.float32)

            init_ops = []

            # Add global_step
            global_step = tf.Variable(0, dtype=tf.int32, trainable=False,
                                  name="global_step")

            if with_privacy:
                gd_op = dp_optimizer.DPGradientDescentOptimizer(
                    lr,
                    [eps, delta],
                    gaussian_sanitizer,
                    sigma=sigma,
                    batches_per_lot=TFFLAGS.batches_per_lot).minimize(
                      cost, global_step=global_step)
            else:
                gd_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)

            saver = tf.train.Saver()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            # We need to maintain the intialization sequence.
            #for v in tf.trainable_variables():
            #    sess.run(tf.variables_initializer([v]))
            #sess.run(tf.global_variables_initializer())
            
            ##### NEED DIS TO LOAD MODEL PLS.
            saver.restore(sess, tf.train.latest_checkpoint(save_path))

            sess.run(init_ops)

            start_time = time.time()
            prev_time = start_time
            filename = "client-" + str(self.index) + "-results-0.json"
            log_path = os.path.join(save_path, filename)
            print(log_path)
            target_eps = [float(s) for s in TFFLAGS.target_eps.split(",")]
            max_target_eps = max(target_eps)

            lot_size = TFFLAGS.batches_per_lot * TFFLAGS.batch_size
            lots_per_epoch = self.num_training_images / lot_size
            for step in range(TFFLAGS.num_training_steps):
                epoch = step / lots_per_epoch
                curr_lr = utils.VaryRate(TFFLAGS.lr, TFFLAGS.end_lr,
                                       TFFLAGS.lr_saturate_epochs, epoch)
                curr_eps = utils.VaryRate(TFFLAGS.eps, TFFLAGS.end_eps,
                                        TFFLAGS.eps_saturate_epochs, epoch)
                for _ in range(TFFLAGS.batches_per_lot):
                    _ = sess.run(
                        [gd_op], feed_dict={lr: curr_lr, eps: curr_eps, delta: TFFLAGS.delta})
                sys.stderr.write("step: %d\n" % step)

                # See if we should stop training due to exceeded privacy budget:
                should_terminate = False
                terminate_spent_eps_delta = None
                if with_privacy and TFFLAGS.terminate_based_on_privacy:
                    terminate_spent_eps_delta = priv_accountant.get_privacy_spent(
                        sess, target_eps=[max_target_eps])[0]
                    # For the Moments accountant, we should always have
                    # spent_eps == max_target_eps.
                    if (terminate_spent_eps_delta.spent_delta > TFFLAGS.target_delta or
                        terminate_spent_eps_delta.spent_eps > max_target_eps):
                      should_terminate = True

                if (eval_steps > 0 and (step + 1) % eval_steps == 0) or should_terminate:
                    if with_privacy:
                      spent_eps_deltas = priv_accountant.get_privacy_spent(
                          sess, target_eps=target_eps)
                    else:
                      spent_eps_deltas = [accountant.EpsDelta(0, 0)]
                    for spent_eps, spent_delta in spent_eps_deltas:
                      sys.stderr.write("spent privacy: eps %.4f delta %.5g\n" % (
                          spent_eps, spent_delta))
                    path = save_path + "/model%s.ckpt"%self.client
                    saver.save(sess, save_path=path)
                    train_accuracy, _ = self.Eval(cifar_train_file, network_parameters,
                                             num_testing_images=self.num_testing_images,
                                             randomize=True, load_path=TFFLAGS.save_path)
                    sys.stderr.write("train_accuracy: %.2f\n" % train_accuracy)
    
                    curr_time = time.time()
                    elapsed_time = curr_time - prev_time
                    prev_time = curr_time
    
                    self.results.append({"step": step+1,  # Number of lots trained so far.
                                    "elapsed_secs": elapsed_time,
                                    "spent_eps_deltas": spent_eps_deltas,
                                    "train_accuracy": train_accuracy})
                    loginfo = {"elapsed_secs": curr_time-start_time,
                               "spent_eps_deltas": spent_eps_deltas,
                               "train_accuracy": train_accuracy,
                               "num_training_steps": step+1,  # Steps so far.
                               "result_series": self.results}
                    loginfo.update(params)
                    if log_path:
                      with tf.gfile.Open(log_path, "w") as f:
                        json.dump(loginfo, f, indent=2)
                        f.write("\n")
                        f.close()
                if should_terminate:
                    for t in tf.trainable_variables():
                        weights.append(t.eval(session=sess))
                    break
                for t in tf.trainable_variables():
                    weights.append(t.eval(session=sess))
            
            coord.request_stop()
            coord.join(threads)

    def Eval(self, cifar_eval_data, network_parameters, num_testing_images,
             randomize, load_path, save_mistakes=False):
      """Evaluate CIFAR-10.

      Args:
        cifar_eval_data: Path of a file containing the CIFAR-10 images to process.
        network_parameters: parameters for defining and training the network.
        num_testing_images: the number of images we will evaluate on.
        randomize: if false, randomize; otherwise, read the testing images
          sequentially.
        load_path: path where to load trained parameters from.
        save_mistakes: save the mistakes if True.

      Returns:
        The evaluation accuracy as a float.
      """
      batch_size = 100
      # Like for training, we need a session for executing the TensorFlow graph.
      with tf.Graph().as_default(), tf.Session() as sess:
        # Create the basic CIFAR model.
        images, labels = cifar10_input.inputs(True, "./data/clients/cifar/", TFFLAGS.batch_size, client=self.client,dp=True)#cifar10_input.distorted_inputs("data/clients/cifar", self.index, TFFLAGS.batch_size)
        logits, _, _ = utils.BuildNetwork(images, network_parameters)
        softmax = tf.nn.softmax(logits)

        # Load the variables.
        ckpt_state = tf.train.get_checkpoint_state(load_path)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
          raise ValueError("No model checkpoint to eval at %s\n" % load_path)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_examples = 0
        correct_predictions = 0
        image_index = 0
        mistakes = []
        for _ in range((num_testing_images + batch_size - 1) // batch_size):
          predictions, label_values = sess.run([softmax, labels])

          # Count how many were predicted correctly.
          for prediction, label_value in zip(predictions, label_values):
            total_examples += 1
            if np.argmax(prediction) == label_value:
              correct_predictions += 1
            elif save_mistakes:
              mistakes.append({"index": image_index,
                               "label": label_value,
                               "pred": np.argmax(prediction)})
            image_index += 1
        coord.request_stop()
        coord.join(threads)

      acc = float(correct_predictions) / float(total_examples)
      print("Client ", self.index, " training accuracy: ", acc)
      return (acc,
              mistakes if save_mistakes else None)

def buildNetwork():
    network_parameters = utils.NetworkParameters()
    
    # If the ASCII proto isn't specified, then construct a config protobuf based
    # on 3 flags.
    network_parameters.input_size = 1 * (IMAGE_SIZE ** 2)
    network_parameters.default_gradient_l2norm_bound = (
        FLAGS.default_gradient_l2norm_bound)
    
    conv = utils.ConvParameters()
    conv.name = "conv1"
    conv.in_channels = 3
    conv.out_channels = 64
    conv.num_outputs = (IMAGE_SIZE // 2) * (IMAGE_SIZE // 2)  * 64
    conv.in_size = IMAGE_SIZE
    #conv.trainable = True
    network_parameters.conv_parameters.append(conv)
    
    conv = utils.ConvParameters()
    conv.name = "conv2"
    conv.in_channels = 64
    conv.out_channels = 64
    conv.num_outputs = (IMAGE_SIZE // 4) * (IMAGE_SIZE // 4) * 64
    conv.in_size = IMAGE_SIZE // 2
    #conv.trainable = True
    network_parameters.conv_parameters.append(conv)
    
    for i in range(FLAGS.num_hidden_layers):
      hidden = utils.LayerParameters()
      hidden.name = "hidden%d" % i
      hidden.num_units = FLAGS.hidden_layer_num_units
      hidden.relu = True
      hidden.with_bias = True
      hidden.trainable = not FLAGS.freeze_bottom_layers
      network_parameters.layer_parameters.append(hidden)
    
    logits = utils.LayerParameters()
    logits.name = "logits"
    logits.num_units = 10
    logits.relu = False
    logits.with_bias = False
    network_parameters.layer_parameters.append(logits)
    network_model = network_parameters

    return network_parameters

def Evaluate(load_path, eval_dir, network_parameters=None, num_testing_images=10000,
             save_mistakes=False):
      """Evaluate CIFAR-10.

      Args:
        cifar_eval_data: Path of a file containing the CIFAR-10 images to process.
        network_parameters: parameters for defining and training the network.
        num_testing_images: the number of images we will evaluate on.
        randomize: if false, randomize; otherwise, read the testing images
          sequentially.
        load_path: path where to load trained parameters from.
        save_mistakes: save the mistakes if True.

      Returns:
        The evaluation accuracy as a float.
      """
      network_parameters=buildNetwork()
      batch_size = 100
      # Like for training, we need a session for executing the TensorFlow graph.
      with tf.Graph().as_default(), tf.Session() as sess:
        # Create the basic Mnist model.
        images, labels = cifar10_input.inputs(True, "./data/cifar10-batches-bin", TFFLAGS.batch_size)
        logits, _, _ = utils.BuildNetwork(images, network_parameters)
        softmax = tf.nn.softmax(logits)
        loss = cifar10.loss(logits, labels)                                                                                                             

        # Load the variables.
        ckpt_state = tf.train.get_checkpoint_state(load_path)
        if not (ckpt_state and ckpt_state.model_checkpoint_path):
          raise ValueError("No model checkpoint to eval at %s\n" % load_path)

        saver = tf.train.Saver()
        saver.restore(sess, ckpt_state.model_checkpoint_path)
        coord = tf.train.Coordinator()
        _ = tf.train.start_queue_runners(sess=sess, coord=coord)

        total_examples = 0
        correct_predictions = 0
        image_index = 0
        mistakes = []
        for _ in range((num_testing_images + batch_size - 1) // batch_size):
          predictions, label_values = sess.run([softmax, labels])

          # Count how many were predicted correctly.
          for prediction, label_value in zip(predictions, label_values):
            total_examples += 1
            if np.argmax(prediction) == label_value:
              correct_predictions += 1
            elif save_mistakes:
              mistakes.append({"index": image_index,
                               "label": label_value,
                               "pred": np.argmax(prediction)})
            image_index += 1
      coord.request_stop()
      acc = float(correct_predictions) / float(total_examples)
      print("Test accuracy: ", acc)
      write(eval_dir, acc)
      return (acc,
              mistakes if save_mistakes else None)  
      
def write(eval_dir, acc):
    filename = eval_dir + "accuracy.pkl"
    old = []            
    if os.path.exists(filename):
        with open(filename,'rb') as rfp: 
            old = pickle.load(rfp, encoding='latin1')
    old.append(acc)
    #old[1].append(acc)                                                                                                                             
    with open(filename, "wb") as fp:   #Pickling
        pickle.dump(old, fp)
        fp.close()
