# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
import warnings
warnings.filterwarnings("ignore")
import tensorflow as tf
import socket
# from reco_utils.recommender.deeprec.models.sequential.sequential_base_model import SequentialBaseModel
from utils.rnn_cell_implement import VecAttGRUCell
from utils.rnn_dien import dynamic_rnn as dynamic_rnn_dien
from tensorflow.keras import backend as K
import abc
import time
import numpy as np
from tensorflow import keras
import os
from tqdm import tqdm

# from reco_utils.recommender.deeprec.models.base_model import BaseModel
from utils.deeprec_utils import cal_metric, cal_weighted_metric, cal_mean_alpha_metric, load_dict

__all__ = ["SURGEModel"]

class BaseModel:
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all deeprec models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.seed = seed
        tf.set_random_seed(seed)
        np.random.seed(seed)

        self.graph = graph if graph is not None else tf.Graph()
        
        self.iterator = iterator_creator(hparams, self.graph)
        self.train_num_ngs = (
            hparams.train_num_ngs if "train_num_ngs" in hparams else None
        )

        with self.graph.as_default():
            self.hparams = hparams

            self.layer_params = []
            self.embed_params = []
            self.cross_params = []
            self.layer_keeps = tf.placeholder(tf.float32, name="layer_keeps")
            self.keep_prob_train = None
            self.keep_prob_test = None
            self.is_train_stage = tf.placeholder(
                tf.bool, shape=(), name="is_training"
            )
            self.group = tf.placeholder(tf.int32, shape=(), name="group")

            self.initializer = self._get_initializer()
            self.logit = self._build_graph()
            self.pred = self._get_pred(self.logit, self.hparams.method)
            self.loss = self._get_loss()
            self.saver = tf.train.Saver(max_to_keep=self.hparams.epochs)
            self.update = self._build_train_opt()
            self.extra_update_ops = tf.get_collection(
                tf.GraphKeys.UPDATE_OPS
            )
            self.init_op = tf.global_variables_initializer()
            self.merged = self._add_summaries()

        # set GPU use with demand growth
        gpu_options = tf.GPUOptions(allow_growth=True)
        self.sess = tf.Session(
            graph=self.graph, config=tf.ConfigProto(gpu_options=gpu_options)
        )
        #  self.sess = tfdbg.LocalCLIDebugWrapperSession(self.sess)
        self.sess.run(self.init_op)

    @abc.abstractmethod
    def _build_graph(self):
        """Subclass will implement this."""
        pass

    def _get_loss(self):
        """Make loss function, consists of data loss and regularization loss
        
        Returns:
            obj: Loss value
        """
        self.data_loss = self._compute_data_loss()
        self.regular_loss = self._compute_regular_loss()
        self.loss = tf.add(self.data_loss, self.regular_loss)
        return self.loss

    def _get_pred(self, logit, task):
        """Make final output as prediction score, according to different tasks.
        
        Args:
            logit (obj): Base prediction value.
            task (str): A task (values: regression/classification)
        
        Returns:
            obj: Transformed score
        """
        if task == "regression":
            pred = tf.identity(logit)
        elif task == "classification":
            pred = tf.sigmoid(logit)
        elif task == 'classifiction_bpr':
            logit_ones, logit_zeros = self.logit
            pred = tf.where(self.iterator.labels > 0, logit_ones - logit_zeros, logit_zeros - logit_ones)
        else:
            raise ValueError(
                "method must be regression or classification, but now is {0}".format(
                    task
                )
            )
        return pred

    def _add_summaries(self):
        tf.summary.scalar("data_loss", self.data_loss)
        tf.summary.scalar("regular_loss", self.regular_loss)
        tf.summary.scalar("loss", self.loss)
        merged = tf.summary.merge_all()
        return merged

    def _l2_loss(self):
        l2_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.embed_l2, tf.nn.l2_loss(param))
            )
        params = self.layer_params
        for param in params:
            l2_loss = tf.add(
                l2_loss, tf.multiply(self.hparams.layer_l2, tf.nn.l2_loss(param))
            )
        return l2_loss

    def _l1_loss(self):
        l1_loss = tf.zeros([1], dtype=tf.float32)
        # embedding_layer l2 loss
        for param in self.embed_params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.embed_l1, tf.norm(param, ord=1))
            )
        params = self.layer_params
        for param in params:
            l1_loss = tf.add(
                l1_loss, tf.multiply(self.hparams.layer_l1, tf.norm(param, ord=1))
            )
        return l1_loss

    def _cross_l_loss(self):
        """Construct L1-norm and L2-norm on cross network parameters for loss function.
        Returns:
            obj: Regular loss value on cross network parameters.
        """
        cross_l_loss = tf.zeros([1], dtype=tf.float32)
        for param in self.cross_params:
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l1, tf.norm(param, ord=1))
            )
            cross_l_loss = tf.add(
                cross_l_loss, tf.multiply(self.hparams.cross_l2, tf.norm(param, ord=2))
            )
        return cross_l_loss

    def _get_initializer(self):
        if self.hparams.init_method == "tnormal":
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "uniform":
            return tf.random_uniform_initializer(
                -self.hparams.init_value, self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "normal":
            return tf.random_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )
        elif self.hparams.init_method == "xavier_normal":
            return tf.contrib.layers.xavier_initializer(uniform=False, seed=self.seed)
        elif self.hparams.init_method == "xavier_uniform":
            return tf.contrib.layers.xavier_initializer(uniform=True, seed=self.seed)
        elif self.hparams.init_method == "he_normal":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=False, seed=self.seed
            )
        elif self.hparams.init_method == "he_uniform":
            return tf.contrib.layers.variance_scaling_initializer(
                factor=2.0, mode="FAN_IN", uniform=True, seed=self.seed
            )
        else:
            return tf.truncated_normal_initializer(
                stddev=self.hparams.init_value, seed=self.seed
            )

    def _compute_data_loss(self):
        if self.hparams.loss == "cross_entropy_loss":
            # import ipdb; ipdb.set_trace()
            data_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=tf.reshape(self.logit, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "square_loss":
            data_loss = tf.sqrt(
                tf.reduce_mean(
                    tf.math.squared_difference(
                        tf.reshape(self.pred, [-1]),
                        tf.reshape(self.iterator.labels, [-1]),
                    )
                )
            )
        elif self.hparams.loss == "log_loss":
            data_loss = tf.reduce_mean(
                tf.losses.log_loss(
                    predictions=tf.reshape(self.pred, [-1]),
                    labels=tf.reshape(self.iterator.labels, [-1]),
                )
            )
        elif self.hparams.loss == "softmax":
            group = self.train_num_ngs + 1
            logits = tf.reshape(self.logit, (-1, group))
            if self.hparams.model_type == "NextItNet":
                labels = (
                    tf.transpose(
                        tf.reshape(
                            self.iterator.labels,
                            (-1, group, self.hparams.max_seq_length),
                        ),
                        [0, 2, 1],
                    ),
                )
                labels = tf.reshape(labels, (-1, group))
            else:
                labels = tf.reshape(self.iterator.labels, (-1, group))
            softmax_pred = tf.nn.softmax(logits, axis=-1)
            boolean_mask = tf.equal(labels, tf.ones_like(labels))
            mask_paddings = tf.ones_like(softmax_pred)
            pos_softmax = tf.where(boolean_mask, softmax_pred, mask_paddings)
            data_loss = -group * tf.reduce_mean(tf.math.log(pos_softmax))
        elif self.hparams.loss == "bpr_loss":
            data_loss = -tf.reduce_mean(tf.log(tf.sigmoid(self.pred)))
        else:
            raise ValueError("this loss not defined {0}".format(self.hparams.loss))
        return data_loss

    def _compute_regular_loss(self):
        """Construct regular loss. Usually it's comprised of l1 and l2 norm.
        Users can designate which norm to be included via config file.
        Returns:
            obj: Regular loss.
        """
        regular_loss = self._l2_loss() + self._l1_loss() + self._cross_l_loss()
        return tf.reduce_sum(regular_loss)

    def _train_opt(self):
        """Get the optimizer according to configuration. Usually we will use Adam.
        Returns:
            obj: An optimizer.
        """
        lr = self.hparams.learning_rate
        optimizer = self.hparams.optimizer

        if optimizer == "adadelta":
            train_step = tf.train.AdadeltaOptimizer(lr)
        elif optimizer == "adagrad":
            train_step = tf.train.AdagradOptimizer(lr)
        elif optimizer == "sgd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "adam":
            train_step = tf.train.AdamOptimizer(lr)
        elif optimizer == "ftrl":
            train_step = tf.train.FtrlOptimizer(lr)
        elif optimizer == "gd":
            train_step = tf.train.GradientDescentOptimizer(lr)
        elif optimizer == "padagrad":
            train_step = tf.train.ProximalAdagradOptimizer(lr)
        elif optimizer == "pgd":
            train_step = tf.train.ProximalGradientDescentOptimizer(lr)
        elif optimizer == "rmsprop":
            train_step = tf.train.RMSPropOptimizer(lr)
        elif optimizer == "lazyadam":
            train_step = tf.contrib.opt.LazyAdamOptimizer(lr)
        else:
            train_step = tf.train.GradientDescentOptimizer(lr)
        return train_step

    def _build_train_opt(self):
        """Construct gradient descent based optimization step
        In this step, we provide gradient clipping option. Sometimes we what to clip the gradients
        when their absolute values are too large to avoid gradient explosion.
        Returns:
            obj: An operation that applies the specified optimization step.
        """
        train_step = self._train_opt()
        gradients, variables = zip(*train_step.compute_gradients(self.loss))
        if self.hparams.is_clip_norm:
            gradients = [
                None
                if gradient is None
                else tf.clip_by_norm(gradient, self.hparams.max_grad_norm)
                for gradient in gradients
            ]
        return train_step.apply_gradients(zip(gradients, variables))

    def _active_layer(self, logit, activation, layer_idx=-1):
        """Transform the input value with an activation. May use dropout.
        
        Args:
            logit (obj): Input value.
            activation (str): A string indicating the type of activation function.
            layer_idx (int): Index of current layer. Used to retrieve corresponding parameters
        
        Returns:
            obj: A tensor after applying activation function on logit.
        """
        if layer_idx >= 0 and self.hparams.user_dropout:
            logit = self._dropout(logit, self.layer_keeps[layer_idx])
        return self._activate(logit, activation, layer_idx)

    def _activate(self, logit, activation, layer_idx=-1):
        if activation == "sigmoid":
            return tf.nn.sigmoid(logit)
        elif activation == "softmax":
            return tf.nn.softmax(logit)
        elif activation == "relu":
            return tf.nn.relu(logit)
        elif activation == "tanh":
            return tf.nn.tanh(logit)
        elif activation == "elu":
            return tf.nn.elu(logit)
        elif activation == "identity":
            return tf.identity(logit)
        elif activation == 'dice':
            return dice(logit, name='dice_{}'.format(layer_idx))
        else:
            raise ValueError("this activations not defined {0}".format(activation))

    def _dropout(self, logit, keep_prob):
        """Apply drops upon the input value.
        Args:
            logit (obj): The input value.
            keep_prob (float): The probability of keeping each element.

        Returns:
            obj: A tensor of the same shape of logit.
        """
        return tf.nn.dropout(x=logit, keep_prob=keep_prob)

    def train(self, sess, feed_dict):
        """Go through the optimization step once with training data in feed_dict.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values to train the model. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of values, including update operation, total loss, data loss, and merged summary.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_train
        feed_dict[self.is_train_stage] = True
        return sess.run(
            [
                self.update,
                self.extra_update_ops,
                self.loss,
                self.data_loss,
                self.merged,
                #  self.pp1,
                #  self.pp2,
                #  self.pp3,
                #  self.pp4,
                #  self.pp5,
                #  self.pp6,
            ],
            feed_dict=feed_dict,
        )

    def eval(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred, self.iterator.labels], feed_dict=feed_dict)

    def infer(self, sess, feed_dict):
        """Given feature data (in feed_dict), get predicted scores with current model.
        Args:
            sess (obj): The model session object.
            feed_dict (dict): Instances to predict. This is a dictionary that maps graph elements to values.

        Returns:
            list: Predicted scores for the given instances.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.pred], feed_dict=feed_dict)

    def load_model(self, model_path=None):
        """Load an existing model.

        Args:
            model_path: model path.

        Raises:
            IOError: if the restore operation failed.
        """
        act_path = self.hparams.load_saved_model
        if model_path is not None:
            act_path = model_path

        try:
            self.saver.restore(self.sess, act_path)
        except:
            raise IOError("Failed to find any matching files for {0}".format(act_path))

    def fit(self, train_file, valid_file, test_file=None):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            test_file (str): test set.

        Returns:
            obj: An instance of self.
        """
        if self.hparams.write_tfevents:
            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        train_sess = self.sess
        for epoch in range(1, self.hparams.epochs + 1):
            step = 0
            self.hparams.current_epoch = epoch

            epoch_loss = 0
            train_start = time.time()
            for (
                batch_data_input,
                impression,
                data_size,
            ) in self.iterator.load_data_from_file(train_file):
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                if self.hparams.write_tfevents:
                    self.writer.add_summary(summary, step)
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )

            train_end = time.time()
            train_time = train_end - train_start

            if self.hparams.save_model:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if epoch % self.hparams.save_epoch == 0:
                    save_path_str = join(self.hparams.MODEL_DIR, "epoch_" + str(epoch))
                    checkpoint_path = self.saver.save(
                        sess=train_sess, save_path=save_path_str
                    )

            eval_start = time.time()
            eval_res = self.run_eval(valid_file)
            train_info = ",".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in [("logloss loss", epoch_loss / step)]
                ]
            )
            eval_info = ", ".join(
                [
                    str(item[0]) + ":" + str(item[1])
                    for item in sorted(eval_res.items(), key=lambda x: x[0])
                ]
            )
            if test_file is not None:
                test_res = self.run_eval(test_file)
                test_info = ", ".join(
                    [
                        str(item[0]) + ":" + str(item[1])
                        for item in sorted(test_res.items(), key=lambda x: x[0])
                    ]
                )
            eval_end = time.time()
            eval_time = eval_end - eval_start

            if test_file is not None:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                    + "\ntest info: "
                    + test_info
                )
            else:
                print(
                    "at epoch {0:d}".format(epoch)
                    + "\ntrain info: "
                    + train_info
                    + "\neval info: "
                    + eval_info
                )
            print(
                "at epoch {0:d} , train time: {1:.1f} eval time: {2:.1f}".format(
                    epoch, train_time, eval_time
                )
            )

        if self.hparams.write_tfevents:
            self.writer.close()

        return self

    def group_labels(self, labels, preds, group_keys):
        """Devide labels and preds into several group according to values in group keys.
        Args:
            labels (list): ground truth label list.
            preds (list): prediction score list.
            group_keys (list): group key list.
        Returns:
            all_labels: labels after group.
            all_preds: preds after group.
        """
        all_keys = list(set(group_keys))
        group_labels = {k: [] for k in all_keys}
        group_preds = {k: [] for k in all_keys}
        for l, p, k in zip(labels, preds, group_keys):
            group_labels[k].append(l)
            group_preds[k].append(p)
        all_labels = []
        all_preds = []
        for k in all_keys:
            all_labels.append(group_labels[k])
            all_preds.append(group_preds[k])
        return all_labels, all_preds

    def run_eval(self, filename):
        """Evaluate the given file and returns some evaluation metrics.

        Args:
            filename (str): A file name that will be evaluated.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """
        load_sess = self.sess
        preds = []
        labels = []
        imp_indexs = []
        for batch_data_input, imp_index, data_size in self.iterator.load_data_from_file(
            filename
        ):
            step_pred, step_labels = self.eval(load_sess, batch_data_input)
            preds.extend(np.reshape(step_pred, -1))
            labels.extend(np.reshape(step_labels, -1))
            imp_indexs.extend(np.reshape(imp_index, -1))
        res = cal_metric(labels, preds, self.hparams.metrics)
        if self.hparams.pairwise_metrics is not None:
            group_labels, group_preds = self.group_labels(labels, preds, imp_indexs)
            res_pairwise = cal_metric(
                group_labels, group_preds, self.hparams.pairwise_metrics
            )
            res.update(res_pairwise)
        return res

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name, format is same as train/val/test file.
            outfile_name (str): Output file name, each line is the predict score.

        Returns:
            obj: An instance of self.
        """
        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input, _, data_size in self.iterator.load_data_from_file(
                infile_name
            ):
                step_pred = self.infer(load_sess, batch_data_input)
                step_pred = step_pred[0][:data_size]
                step_pred = np.reshape(step_pred, -1)
                wt.write("\n".join(map(str, step_pred)))
                # line break after each batch.
                wt.write("\n")
        return self

    def _attention(self, inputs, attention_size):
        """Soft alignment attention implement.
        
        Args:
            inputs (obj): Sequences ready to apply attention.
            attention_size (int): The dimension of attention operation.

        Returns:
            obj: Weighted sum after attention.
        """
        hidden_size = inputs.shape[2].value
        if not attention_size:
            attention_size = hidden_size

        attention_mat = tf.get_variable(
            name="attention_mat",
            shape=[inputs.shape[-1].value, hidden_size],
            initializer=self.initializer,
        )
        att_inputs = tf.tensordot(inputs, attention_mat, [[2], [0]])

        query = tf.get_variable(
            name="query",
            shape=[attention_size],
            dtype=tf.float32,
            initializer=self.initializer,
        )
        att_logits = tf.tensordot(att_inputs, query, axes=1, name="att_logits")
        att_weights = tf.nn.softmax(att_logits, name="att_weights")
        output = inputs * tf.expand_dims(att_weights, -1)
        return output

    def _fcn_net(self, model_output, layer_sizes, scope, reuse=False): # add reuse, for BPR loss
        """Construct the MLP part for the model.

        Args:
            model_output (obj): The output of upper layers, input of MLP part
            layer_sizes (list): The shape of each layer of MLP part
            scope (obj): The scope of MLP part

        Returns:s
            obj: prediction logit after fully connected layer
        """
        hparams = self.hparams
        with tf.variable_scope(scope, reuse=reuse):
            last_layer_size = model_output.shape[-1]
            layer_idx = 0
            hidden_nn_layers = []
            hidden_nn_layers.append(model_output)
            with tf.variable_scope("nn_part", initializer=self.initializer) as scope:
                for idx, layer_size in enumerate(layer_sizes):
                    curr_w_nn_layer = tf.get_variable(
                        name="w_nn_layer" + str(layer_idx),
                        shape=[last_layer_size, layer_size],
                        dtype=tf.float32
                    )
                    curr_b_nn_layer = tf.get_variable(
                        name="b_nn_layer" + str(layer_idx),
                        shape=[layer_size],
                        dtype=tf.float32,
                        initializer=tf.zeros_initializer(),
                    )
                    tf.summary.histogram(
                        "nn_part/" + "w_nn_layer" + str(layer_idx), curr_w_nn_layer
                    )
                    tf.summary.histogram(
                        "nn_part/" + "b_nn_layer" + str(layer_idx), curr_b_nn_layer
                    )
                    curr_hidden_nn_layer = (
                        tf.tensordot(
                            hidden_nn_layers[layer_idx], curr_w_nn_layer, axes=1
                        )
                        + curr_b_nn_layer
                    )

                    scope = "nn_part" + str(idx)
                    activation = hparams.activation[idx]

                    if hparams.enable_BN is True:
                        curr_hidden_nn_layer = tf.layers.batch_normalization(
                            curr_hidden_nn_layer,
                            momentum=0.95,
                            epsilon=0.0001,
                            training=self.is_train_stage,
                        )

                    curr_hidden_nn_layer = self._active_layer(
                        logit=curr_hidden_nn_layer, activation=activation, layer_idx=idx
                    )
                    hidden_nn_layers.append(curr_hidden_nn_layer)
                    layer_idx += 1
                    last_layer_size = layer_size

                w_nn_output = tf.get_variable(
                    name="w_nn_output", shape=[last_layer_size, 1], dtype=tf.float32
                )
                b_nn_output = tf.get_variable(
                    name="b_nn_output",
                    shape=[1],
                    dtype=tf.float32,
                    initializer=tf.zeros_initializer(),
                )
                tf.summary.histogram(
                    "nn_part/" + "w_nn_output" + str(layer_idx), w_nn_output
                )
                tf.summary.histogram(
                    "nn_part/" + "b_nn_output" + str(layer_idx), b_nn_output
                )
                nn_output = (
                    tf.tensordot(hidden_nn_layers[-1], w_nn_output, axes=1)
                    + b_nn_output
                )
                self.logit = nn_output
                return nn_output



class SequentialBaseModel(BaseModel):
    def __init__(self, hparams, iterator_creator, graph=None, seed=None):
        """Initializing the model. Create common logics which are needed by all sequential models, such as loss function, 
        parameter set.

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
            graph (obj): An optional graph.
            seed (int): Random seed.
        """
        self.hparams = hparams

        self.need_sample = hparams.need_sample
        self.train_num_ngs = hparams.train_num_ngs
        if self.train_num_ngs is None:
            raise ValueError(
                "Please confirm the number of negative samples for each positive instance."
            )
        self.min_seq_length = (
            hparams.min_seq_length if "min_seq_length" in hparams else 1
        )
        self.hidden_size = hparams.hidden_size if "hidden_size" in hparams else None
        self.graph = tf.Graph() if not graph else graph

        with self.graph.as_default():
            self.embedding_keeps = tf.placeholder(tf.float32, name="embedding_keeps")
            self.embedding_keep_prob_train = None
            self.embedding_keep_prob_test = None

        super().__init__(hparams, iterator_creator, graph=self.graph, seed=seed)

    @abc.abstractmethod
    def _build_seq_graph(self):
        """Subclass will implement this."""
        pass

    def _build_graph(self):
        """The main function to create sequential models.
        
        Returns:
            obj:the prediction score make by the model.
        """
        hparams = self.hparams
        self.keep_prob_train = 1 - np.array(hparams.dropout)
        self.keep_prob_test = np.ones_like(hparams.dropout)

        self.embedding_keep_prob_train = 1.0 - hparams.embedding_dropout
        if hparams.test_dropout:
            self.embedding_keep_prob_test = 1.0 - hparams.embedding_dropout
        else:
            self.embedding_keep_prob_test = 1.0

        with tf.variable_scope("sequential") as self.sequential_scope:
            self._build_embedding()
            self._lookup_from_embedding()
            model_output = self._build_seq_graph()
            if len(model_output) == 1:
                logit = self._fcn_net(model_output, hparams.layer_sizes, scope="logit_fcn")
            else:
                logit_ones = self._fcn_net(model_output[0], hparams.layer_sizes, scope="logit_fcn")
                logit_zeros = self._fcn_net(model_output[1], hparams.layer_sizes, scope="logit_fcn", reuse=True)
                logit = (logit_ones, logit_zeros)
            self._add_norm()
            return logit

    def train(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_train
        return super(SequentialBaseModel, self).train(sess, feed_dict)

    #  def batch_train(self, file_iterator, train_sess, vm, tb):
    def batch_train(self, file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs, pre_step = 0):
        """Train the model for a single epoch with mini-batches.

        Args:
            file_iterator (Iterator): iterator for training data.
            train_sess (Session): tf session for training.
            vm (VizManager): visualization manager for visdom.
            tb (TensorboardX): visualization manager for TensorboardX.

        Returns:
        epoch_loss: total loss of the single epoch.

        """
        # import ipdb; ipdb.set_trace()
        step = pre_step
        epoch_loss = 0
        # import ipdb; ipdb.set_trace()
        for batch_data_input in tqdm(file_iterator):
            if batch_data_input:
                step_result = self.train(train_sess, batch_data_input)
                (_, _, step_loss, step_data_loss, summary) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _, _, _, _) = step_result
                #  (_, _, step_loss, step_data_loss, summary, _, _, _,) = step_result
                if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
                    self.writer.add_summary(summary, step)
                # import ipdb; ipdb.set_trace()
                epoch_loss += step_loss
                step += 1
                if step % self.hparams.show_step == 0:
                    print(
                        "step {0:d} , total_loss: {1:.4f}, data_loss: {2:.4f}".format(
                            step, step_loss, step_data_loss
                        )
                    )
                    if self.hparams.visual_type == 'epoch':
                        if vm != None:
                            vm.step_update_line('loss', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('loss', step_loss, step)

                if self.hparams.visual_type == 'step':
                    if step % self.hparams.visual_step == 0:
                        if vm != None:
                            vm.step_update_line('loss', step_loss)
                        #  tf.summary.scalar('loss',step_loss)
                        tb.add_scalar('loss', step_loss, step)

                        # steps validation for visualization
                        
                        valid_res = self.run_weighted_eval(valid_file, valid_num_ngs)
                        # self._compute_data_loss()
                        if vm != None:
                            vm.step_update_multi_lines(valid_res)  # TODO
                        for vs in valid_res:
                            #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                            tb.add_scalar(vs.replace('@', '_'), valid_res[vs], step)


        return epoch_loss, step

    def fit(
        self, train_file, valid_file, valid_num_ngs, eval_metric="group_auc", vm=None, tb=None, pretrain=False
    ):
        """Fit the model with train_file. Evaluate the model on valid_file per epoch to observe the training status.
        If test_file is not None, evaluate it too.
        
        Args:
            train_file (str): training data set.
            valid_file (str): validation set.
            valid_num_ngs (int): the number of negative instances with one positive instance in validation data.
            eval_metric (str): the metric that control early stopping. e.g. "auc", "group_auc", etc.

        Returns:
            obj: An instance of self.
        """

        # check bad input.
        # import ipdb; ipdb.set_trace()
        # if not self.need_sample and self.train_num_ngs < 1:
        #     raise ValueError(
        #         "Please specify a positive integer of negative numbers for training without sampling needed."
        #     )
        # if valid_num_ngs < 1:
        #     raise ValueError(
        #         "Please specify a positive integer of negative numbers for validation."
        #     )

        if self.need_sample and self.train_num_ngs < 1:
            self.train_num_ngs = 1

        if self.hparams.write_tfevents and self.hparams.SUMMARIES_DIR:
            if not os.path.exists(self.hparams.SUMMARIES_DIR):
                os.makedirs(self.hparams.SUMMARIES_DIR)

            self.writer = tf.summary.FileWriter(
                self.hparams.SUMMARIES_DIR, self.sess.graph
            )

        #  if pretrain:
            #  self.saver_emb = tf.train.Saver({'item_lookup':'item_embedding', 'user_lookup':'user_embedding'},max_to_keep=self.hparams.epochs)
        if pretrain:
            print('start saving embedding')
            if not os.path.exists(self.hparams.PRETRAIN_DIR):
                os.makedirs(self.hparams.PRETRAIN_DIR)
            #  checkpoint_emb_path = self.saver_emb.save(
                #  sess=train_sess,
                #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
            #  )
            #  graph_def = tf.get_default_graph().as_graph_def()
            var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
            constant_graph = tf.graph_util.convert_variables_to_constants(self.sess, self.sess.graph_def, var_list)
            with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                f.write(constant_graph.SerializeToString())
            print('embedding saved')

        train_sess = self.sess
        eval_info = list()

        best_metric, self.best_epoch = 0, 0
        # import ipdb; ipdb.set_trace()
        step = 1
        for epoch in range(1, self.hparams.epochs + 1):
            self.hparams.current_epoch = epoch
            file_iterator = self.iterator.load_data_from_file(
                train_file,
                min_seq_length=self.min_seq_length,
                batch_num_ngs=self.train_num_ngs,
            )
            # import ipdb; ipdb.set_trace()
            #  epoch_loss = self.batch_train(file_iterator, train_sess, vm, tb)
            
            epoch_loss, step = self.batch_train(file_iterator, train_sess, vm, tb, valid_file, valid_num_ngs, step)
            if vm != None:
                vm.step_update_line('epoch loss', epoch_loss)
            #  tf.summary.scalar('epoch loss', epoch_loss)
            tb.add_scalar('epoch_loss', epoch_loss, epoch)

            valid_res = self.run_weighted_eval(valid_file, valid_num_ngs) ### 评估
            print(
                "eval valid at epoch {0}: {1}".format(
                    epoch,
                    ",".join(
                        [
                            "" + str(key) + ":" + str(value)
                            for key, value in valid_res.items()
                        ]
                    ),
                )
            )

            # add train auc, logloss
            # train_res = self.run_weighted_eval(train_file, 0)
            # print(
            #     "eval train at epoch {0}: {1}".format(
            #         epoch,
            #         ",".join(
            #             [
            #                 "" + str(key) + ":" + str(value)
            #                 for key, value in train_res.items()
            #             ]
            #         ),
            #     )
            # )

            if self.hparams.visual_type == 'epoch':
                if vm != None:
                    vm.step_update_multi_lines(valid_res)  # TODO
                for vs in valid_res:
                    #  tf.summary.scalar(vs.replace('@', '_'), valid_res[vs])
                    tb.add_scalar(vs.replace('@', '_'), valid_res[vs], epoch)
                    # tb.add_scalar(f"train_{vs.replace('@', '_')}", train_res[vs], epoch)
            eval_info.append((epoch, valid_res))

            # early stop
            progress = False
            early_stop = self.hparams.EARLY_STOP
            if valid_res[eval_metric] > best_metric:
                best_metric = valid_res[eval_metric]
                self.best_epoch = epoch
                progress = True
            else:
                if early_stop > 0 and epoch - self.best_epoch >= early_stop:
                    print("early stop at epoch {0}!".format(epoch))

                    if pretrain:
                        if not os.path.exists(self.hparams.PRETRAIN_DIR):
                            os.makedirs(self.hparams.PRETRAIN_DIR)
                        #  checkpoint_emb_path = self.saver_emb.save(
                            #  sess=train_sess,
                            #  save_path=self.hparams.PRETRAIN_DIR + "epoch_" + str(epoch),
                        #  )
                        #  graph_def = tf.get_default_graph().as_graph_def()
                        var_list = ['sequential/embedding/item_embedding', 'sequential/embedding/user_embedding']
                        constant_graph = tf.graph_util.convert_variables_to_constants(train_sess, train_sess.graph_def, var_list)
                        with tf.gfile.FastGFile(self.hparams.PRETRAIN_DIR + "test-model.pb", mode='wb') as f:
                            f.write(constant_graph.SerializeToString())

                    break

            if self.hparams.save_model and self.hparams.MODEL_DIR:
                if not os.path.exists(self.hparams.MODEL_DIR):
                    os.makedirs(self.hparams.MODEL_DIR)
                if progress:
                    checkpoint_path = self.saver.save(
                        sess=train_sess,
                        save_path=self.hparams.MODEL_DIR + "epoch_" + str(epoch),
                    )


        if self.hparams.write_tfevents:
            self.writer.close()
        # import ipdb; ipdb.set_trace()
        print(eval_info)
        print("best epoch: {0}".format(self.best_epoch))
        return self

    def run_eval(self, filename, num_ngs):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                step_pred, step_labels = self.eval(load_sess, batch_data_input)
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))

        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        return res

    def eval(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).eval(sess, feed_dict)

    def run_weighted_eval(self, filename, num_ngs, calc_mean_alpha=False):
        """Evaluate the given file and returns some evaluation metrics.
        
        Args:
            filename (str): A file name that will be evaluated.
            num_ngs (int): The number of negative sampling for a positive instance.

        Returns:
            dict: A dictionary contains evaluation metrics.
        """

        load_sess = self.sess
        users = []
        preds = []
        labels = []
        group_preds = []
        group_labels = []
        group = num_ngs + 1 # 一个正例， num_args 个负例
        if calc_mean_alpha:
            alphas = []

        for batch_data_input in self.iterator.load_data_from_file(
            filename, min_seq_length=self.min_seq_length, batch_num_ngs=0
        ):
            if batch_data_input:
                if not calc_mean_alpha: # ok
                    step_user, step_pred, step_labels = self.eval_with_user(load_sess, batch_data_input)
                else:
                    step_user, step_pred, step_labels, step_alpha = self.eval_with_user_and_alpha(load_sess, batch_data_input)
                    alphas.extend(np.reshape(step_alpha, -1))
                users.extend(np.reshape(step_user, -1))
                preds.extend(np.reshape(step_pred, -1))
                labels.extend(np.reshape(step_labels, -1))
                group_preds.extend(np.reshape(step_pred, (-1, group)))
                group_labels.extend(np.reshape(step_labels, (-1, group)))
        # print(preds)
        res = cal_metric(labels, preds, self.hparams.metrics)
        res_pairwise = cal_metric(
            group_labels, group_preds, self.hparams.pairwise_metrics
        )
        res.update(res_pairwise)
        res_weighted = cal_weighted_metric(users, preds, labels, self.hparams.weighted_metrics)
        res.update(res_weighted)
        if calc_mean_alpha:
            res_alpha = cal_mean_alpha_metric(alphas, labels)
            res.update(res_alpha)
        return res

    def eval_with_user(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels], feed_dict=feed_dict)

    def eval_with_user_and_alpha(self, sess, feed_dict):
        """Evaluate the data in feed_dict with current model.

        Args:
            sess (obj): The model session object.
            feed_dict (dict): Feed values for evaluation. This is a dictionary that maps graph elements to values.

        Returns:
            list: A list of evaluated results, including total loss value, data loss value,
                predicted scores, and ground-truth labels.
        """
        feed_dict[self.layer_keeps] = self.keep_prob_test
        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        feed_dict[self.is_train_stage] = False
        return sess.run([self.iterator.users, self.pred, self.iterator.labels, self.alpha_output], feed_dict=feed_dict)

    def predict(self, infile_name, outfile_name):
        """Make predictions on the given data, and output predicted scores to a file.
        
        Args:
            infile_name (str): Input file name.
            outfile_name (str): Output file name.

        Returns:
            obj: An instance of self.
        """

        load_sess = self.sess
        with tf.gfile.GFile(outfile_name, "w") as wt:
            for batch_data_input in self.iterator.load_data_from_file(
                infile_name, batch_num_ngs=0
            ):
                if batch_data_input:
                    step_pred = self.infer(load_sess, batch_data_input)
                    step_pred = np.reshape(step_pred, -1)
                    wt.write("\n".join(map(str, step_pred)))
                    wt.write("\n")
        return self

    def infer(self, sess, feed_dict):

        feed_dict[self.embedding_keeps] = self.embedding_keep_prob_test
        return super(SequentialBaseModel, self).infer(sess, feed_dict)

    def _build_embedding(self):
        """The field embedding layer. Initialization of embedding variables."""
        hparams = self.hparams
        # self.user_vocab_length = len(load_dict(hparams.user_vocab))
        self.item_vocab_length = len(load_dict(hparams.item_vocab))
        self.cate_vocab_length = len(load_dict(hparams.cate_vocab))
        # self.user_embedding_dim = hparams.user_embedding_dim
        self.item_embedding_dim = hparams.item_embedding_dim
        self.cate_embedding_dim = hparams.cate_embedding_dim

        with tf.variable_scope("embedding", initializer=self.initializer):
            # self.user_lookup = tf.get_variable(
            #     name="user_embedding",
            #     shape=[self.user_vocab_length, self.user_embedding_dim],
            #     dtype=tf.float32,
            # )
            self.item_lookup = tf.get_variable(
                name="item_embedding",
                shape=[self.item_vocab_length, self.item_embedding_dim],
                dtype=tf.float32,
            )
            self.cate_lookup = tf.get_variable(
                name="cate_embedding",
                shape=[self.cate_vocab_length, self.cate_embedding_dim],
                dtype=tf.float32,
            )

        print(self.hparams.FINETUNE_DIR)
        print(not self.hparams.FINETUNE_DIR)
        if self.hparams.FINETUNE_DIR:
            # import pdb; pdb.set_trace()
            with tf.Session() as sess:
                # with tf.gfile.FastGFile(output_graph_path, 'rb') as f:
                #     graph_def = tf.GraphDef()
                #     graph_def.ParseFromString(f.read())
                #     sess.graph.as_default()
                #     tf.import_graph_def(graph_def, name='')
                #  tf.global_variables_initializer().run()
                output_graph_def = tf.GraphDef()
                with open(self.hparams.FINETUNE_DIR + "test-model.pb", "rb") as f:
                    output_graph_def.ParseFromString(f.read())
                    _ = tf.import_graph_def(output_graph_def, name="")

                self.item_lookup = sess.graph.get_tensor_by_name('sequential/embedding/item_embedding')
                self.user_lookup = sess.graph.get_tensor_by_name('sequential/embedding/user_embedding')
            #  print(input_x.eval())

            #  output = sess.graph.get_tensor_by_name("conv/b:0")

    def _lookup_from_embedding(self):
        """Lookup from embedding variables. A dropout layer follows lookup operations.
        """
        # self.user_embedding = tf.nn.embedding_lookup(
        #     self.user_lookup, self.iterator.users
        # )
        # tf.summary.histogram("user_embedding_output", self.user_embedding)

        self.item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.items
        )
        self.item_history_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.iterator.item_history
        )
        tf.summary.histogram(
            "item_history_embedding_output", self.item_history_embedding
        )

        self.cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.cates
        )
        self.cate_history_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.iterator.item_cate_history
        )
        tf.summary.histogram(
            "cate_history_embedding_output", self.cate_history_embedding
        )

        involved_items = tf.concat(
            [
                tf.reshape(self.iterator.item_history, [-1]),
                tf.reshape(self.iterator.items, [-1]),
            ],
            -1,
        )
        self.involved_items, _ = tf.unique(involved_items)
        involved_item_embedding = tf.nn.embedding_lookup(
            self.item_lookup, self.involved_items
        )
        self.embed_params.append(involved_item_embedding)

        involved_cates = tf.concat(
            [
                tf.reshape(self.iterator.item_cate_history, [-1]),
                tf.reshape(self.iterator.cates, [-1]),
            ],
            -1,
        )
        self.involved_cates, _ = tf.unique(involved_cates)
        involved_cate_embedding = tf.nn.embedding_lookup(
            self.cate_lookup, self.involved_cates
        )
        self.embed_params.append(involved_cate_embedding)

        # import ipdb; ipdb.set_trace()
        if self.hparams.dataset in ['taobao_global', 'yelp_global', 'taobao_global_small']:
            self.target_item_embedding = tf.concat(
                [self.item_embedding, self.cate_embedding], -1
            )
        else:
            # self.target_item_embedding = self.item_embedding
            self.target_item_embedding = tf.concat(
                [self.item_embedding], -1
            )

        tf.summary.histogram("target_item_embedding_output", self.target_item_embedding)

        # dropout after embedding
        # self.user_embedding = self._dropout(
        #     self.user_embedding, keep_prob=self.embedding_keeps
        # )
        self.item_history_embedding = self._dropout(
            self.item_history_embedding, keep_prob=self.embedding_keeps
        )
        self.cate_history_embedding = self._dropout(
            self.cate_history_embedding, keep_prob=self.embedding_keeps
        )
        self.target_item_embedding = self._dropout(
            self.target_item_embedding, keep_prob=self.embedding_keeps
        )

    def _add_norm(self):
        """Regularization for embedding variables and other variables."""
        all_variables, embed_variables = (
            tf.trainable_variables(),
            tf.trainable_variables(self.sequential_scope._name + "/embedding"),
        )
        layer_params = list(set(all_variables) - set(embed_variables))
        self.layer_params.extend(layer_params)


class SURGEModel(SequentialBaseModel):

    def __init__(self, hparams, iterator_creator, seed=None):
        """Initialization of variables or temp hyperparameters

        Args:
            hparams (obj): A tf.contrib.training.HParams object, hold the entire set of hyperparameters.
            iterator_creator (obj): An iterator to load the data.
        """
        self.hparams = hparams
        self.relative_threshold = 0.5 
        self.metric_heads = 1
        self.attention_heads = 1
        self.pool_layers = 1
        self.layer_shared = True
        if 'kwai' in socket.gethostname():
            self.pool_length = 150 # kuaishou
        else:
            self.pool_length = 30 # taobao
        super().__init__(hparams, iterator_creator, seed=seed)


    def _build_seq_graph(self):
        """ SURGE Model: 

            1) Interest graph: Graph construction based on metric learning
            2) Interest fusion and extraction : Graph convolution and graph pooling 
            3) Prediction: Flatten pooled graph to reduced sequence
        """
        # B: batch size, L: max_len
        # print(f'histroy embedding: {self.item_history_embedding.shape}')
        X = tf.concat(
            [self.item_history_embedding, self.cate_history_embedding], 2
        ) # (bs, seq_len, item_dim), (B, L, ans_dim)
        self.mask = self.iterator.mask # (B, L)
        self.float_mask = tf.cast(self.mask, tf.float32)
        self.real_sequence_length = tf.reduce_sum(self.mask, 1) #(B, )
        # import ipdb; ipdb.set_trace()
        with tf.name_scope('interest_graph'): # construct A
            ## Node similarity metric learning 
            S = []
            for i in range(self.metric_heads):
                # weighted cosine similarity
                self.weighted_tensor = tf.layers.dense(tf.ones([1, 1]), X.shape.as_list()[-1], use_bias=False) #(1, item_dim + ans_dim)
                X_fts = X * tf.expand_dims(self.weighted_tensor, 0) # (B, L, item_dim + ans_dim), w * h
                X_fts = tf.nn.l2_normalize(X_fts,dim=2) 
                S_one = tf.matmul(X_fts, tf.transpose(X_fts, (0,2,1))) # (B, L, L), M_delta (1)
                # min-max normalization for mask
                S_min = tf.reduce_min(S_one, -1, keepdims=True)
                S_max = tf.reduce_max(S_one, -1, keepdims=True)
                S_one = (S_one - S_min) / (S_max - S_min)
                S += [S_one]
            S = tf.reduce_mean(tf.stack(S, 0), 0) # (B, L, L), M (2)
            # mask invalid nodes
            S = S * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)

            ## Graph sparsification via seted sparseness 
            S_flatten = tf.reshape(S, [tf.shape(S)[0],-1]) # （B, L * L)
            if 'kwai' in socket.gethostname():
                sorted_S_flatten = tf.contrib.framework.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_S_flatten = tf.sort(S_flatten, direction='DESCENDING', axis=-1) # B*L -> B*L
            # relative ranking strategy of the entire graph
            num_edges = tf.cast(tf.count_nonzero(S, [1,2]), tf.float32) # B
            to_keep_edge = tf.cast(tf.math.ceil(num_edges * self.relative_threshold), tf.int32)
            if 'kwai' in socket.gethostname():
                threshold_index = tf.stack([tf.range(tf.shape(X)[0]), tf.cast(to_keep_edge, tf.int32)], 1) # B*2
                threshold_score = tf.gather_nd(sorted_S_flatten, threshold_index) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                threshold_score = tf.gather_nd(sorted_S_flatten, tf.expand_dims(tf.cast(to_keep_edge, tf.int32), -1), batch_dims=1) # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            A = tf.cast(tf.greater(S, tf.expand_dims(tf.expand_dims(threshold_score, -1), -1)), tf.float32) # （B, L * L)


        with tf.name_scope('interest_fusion_extraction'):
            for l in range(self.pool_layers):
                reuse = False if l==0 else True
                X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A, layer=l, reuse=reuse)


        with tf.name_scope('prediction'):
            # flatten pooled graph to reduced sequence 
            output_shape = self.mask.get_shape() # self.mask, graph pooling 后
            if 'kwai' in socket.gethostname():
                sorted_mask_index = tf.contrib.framework.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
                sorted_mask = tf.contrib.framework.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
            else:
                sorted_mask_index = tf.argsort(self.mask, direction='DESCENDING', stable=True, axis=-1) # B*L -> B*L
                sorted_mask = tf.sort(self.mask, direction='DESCENDING', axis=-1) # B*L -> B*L
            sorted_mask.set_shape(output_shape)
            sorted_mask_index.set_shape(output_shape)
            X = tf.batch_gather(X, sorted_mask_index) # B*L*F  < B*L = B*L*F
            self.mask = sorted_mask
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B

            # cut useless sequence tail per batch 
            self.to_max_length = tf.range(tf.reduce_max(self.reduced_sequence_length)) # l
            X = tf.gather(X, self.to_max_length, axis=1) # B*L*F -> B*l*F
            self.mask = tf.gather(self.mask, self.to_max_length, axis=1) # B*L -> B*l
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1) # B

            # use cluster score as attention weights in AUGRU 
            _, alphas = self._attention_fcn(self.target_item_embedding, X, 'AGRU', False, return_alpha=True)
            _, final_state = dynamic_rnn_dien(
                VecAttGRUCell(self.hparams.hidden_size),
                inputs=X,
                att_scores = tf.expand_dims(alphas, -1),
                sequence_length=self.reduced_sequence_length,
                dtype=tf.float32,
                scope="gru"
            )
            if self.hparams.dataset in ['taobao_global', 'yelp_global', 'taobao_global_small']:
                model_output = tf.concat([final_state, graph_readout, self.target_item_embedding, graph_readout*self.target_item_embedding], 1)
                return model_output
            else:
                ones_vector = tf.ones_like(self.iterator.cates)
                zeros_vector = tf.zeros_like(self.iterator.cates)
                cate_embedding_ones = tf.nn.embedding_lookup(
                    self.cate_lookup, ones_vector
                )
                cate_embedding_zeros = tf.nn.embedding_lookup(
                    self.cate_lookup, zeros_vector
                )

                target_item_embedding_ones = tf.concat(
                    [self.target_item_embedding, cate_embedding_ones], -1
                )

                target_item_embedding_zeros = tf.concat(
                    [self.target_item_embedding, cate_embedding_zeros], -1
                )

                model_output_ones = tf.concat([final_state, graph_readout, target_item_embedding_ones, graph_readout*target_item_embedding_ones], 1)
                model_output_zeros = tf.concat([final_state, graph_readout, target_item_embedding_zeros, graph_readout*target_item_embedding_zeros], 1)
                # import ipdb; ipdb.set_trace()
                return model_output_ones, model_output_zeros
  
    def _attention_fcn(self, query, key_value, name, reuse, return_alpha=False):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            name (obj): The name of variable W 
            reuse (obj): Reusing variable W in query operation 
            return_alpha (obj): Returning attention weights

        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        with tf.variable_scope("attention_fcn"+str(name), reuse=reuse):
            query_size = query.shape[-1].value
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))

            attention_mat = tf.get_variable(
                name="attention_mat"+str(name),
                shape=[key_value.shape.as_list()[-1], query_size],
                initializer=self.initializer,
            ) # W_c
            att_inputs = tf.tensordot(key_value, attention_mat, [[2], [0]])

            if query.shape.ndims != att_inputs.shape.ndims:
                queries = tf.reshape(
                    tf.tile(query, [1, tf.shape(att_inputs)[1]]), tf.shape(att_inputs)
                )
            else:
                queries = query

            last_hidden_nn_layer = tf.concat(
                [att_inputs, queries, att_inputs - queries, att_inputs * queries], -1
            )
            att_fnc_output = self._fcn_net(
                last_hidden_nn_layer, self.hparams.att_fcn_layer_sizes, scope="att_fcn"
            )
            att_fnc_output = tf.squeeze(att_fnc_output, -1) # 公式（6）
            mask_paddings = tf.ones_like(att_fnc_output) * (-(2 ** 32) + 1)
            att_weights = tf.nn.softmax(
                tf.where(boolean_mask, att_fnc_output, mask_paddings),
                name="att_weights",
            )
            output = key_value * tf.expand_dims(att_weights, -1)
            if not return_alpha:
                return output
            else:
                return output, att_weights


    def _interest_fusion_extraction(self, X, A, layer, reuse):
        """Interest fusion and extraction via graph convolution and graph pooling 

        Args:
            X (obj): Node embedding of graph
            A (obj): Adjacency matrix of graph
            layer (obj): Interest fusion and extraction layer
            reuse (obj): Reusing variable W in query operation 

        Returns:
            X (obj): Aggerated cluster embedding 
            A (obj): Pooled adjacency matrix 
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer

        """
        
        with tf.name_scope('interest_fusion'): # 3.2.2
            ## cluster embedding
            A_bool = tf.cast(tf.greater(A, 0), A.dtype)
            A_bool = A_bool * (tf.ones([A.shape.as_list()[1],A.shape.as_list()[1]]) - tf.eye(A.shape.as_list()[1])) + tf.eye(A.shape.as_list()[1]) #  sets the diagonal elements of A_bool to one 
            D = tf.reduce_sum(A_bool, axis=-1) # (B, L)
            D = tf.sqrt(D)[:, None] + K.epsilon() # (B, 1, L)
            A = (A_bool / D) / tf.transpose(D, perm=(0,2,1)) # B*L*L / B*1*L / B*L*1
            X_q = tf.matmul(A, tf.matmul(A, X)) # (B, L, F), 2 GCN

            Xc = []
            for i in range(self.attention_heads):
                ## cluster- and query-aware attention
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_layer_'+str(layer)+'_'+str(i), False, return_alpha=True)
                if self.layer_shared: # ok
                    _, f_1 = self._attention_fcn(X_q, X, 'f1_shared'+'_'+str(i), reuse, return_alpha=True)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, X, 'f2_shared'+'_'+str(i), reuse, return_alpha=True)

                ## graph attentive convolution
                E = A_bool * tf.expand_dims(f_1,1) + A_bool * tf.transpose(tf.expand_dims(f_2,1), (0,2,1)) # B*L*1 x B*L*1 -> B*L*L
                E = tf.nn.leaky_relu(E)
                boolean_mask = tf.equal(A_bool, tf.ones_like(A_bool))
                mask_paddings = tf.ones_like(E) * (-(2 ** 32) + 1)
                E = tf.nn.softmax(
                    tf.where(boolean_mask, E, mask_paddings),
                    axis = -1
                )
                Xc_one = tf.matmul(E, X) # B*L*L x B*L*F -> B*L*F
                
                Xc_one = tf.layers.dense(Xc_one, self.item_embedding_dim+self.cate_embedding_dim, use_bias=False)
                # import ipdb; ipdb.set_trace()
                Xc_one += X
                Xc += [tf.nn.leaky_relu(Xc_one)]
            Xc = tf.reduce_mean(tf.stack(Xc, 0), 0) #(B, L, F)

        with tf.name_scope('interest_extraction'): # 3.3
            ## cluster fitness score 
            X_q = tf.matmul(A, tf.matmul(A, Xc)) # B*L*F
            cluster_score = []
            for i in range(self.attention_heads):
                if not self.layer_shared:
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_layer_'+str(layer)+'_'+str(i), True, return_alpha=True) 
                    _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_layer_'+str(layer)+'_'+str(i), True, return_alpha=True)
                if self.layer_shared:
                    # import ipdb; ipdb.set_trace()
                    _, f_1 = self._attention_fcn(X_q, Xc, 'f1_shared'+'_'+str(i), True, return_alpha=True) #, (B, L)
                    _, f_2 = self._attention_fcn(self.target_item_embedding, Xc, 'f2_shared'+'_'+str(i), True, return_alpha=True) #, (B, L)
                cluster_score += [f_1 + f_2]
            cluster_score = tf.reduce_mean(tf.stack(cluster_score, 0), 0) # (B, L)
            boolean_mask = tf.equal(self.mask, tf.ones_like(self.mask))
            mask_paddings = tf.ones_like(cluster_score) * (-(2 ** 32) + 1)
            cluster_score = tf.nn.softmax(
                tf.where(boolean_mask, cluster_score, mask_paddings),
                axis = -1
            )

            ## graph pooling
            num_nodes = tf.reduce_sum(self.mask, 1) # B
            boolean_pool = tf.greater(num_nodes, self.pool_length)
            to_keep = tf.where(boolean_pool, 
                               tf.cast(self.pool_length + (self.real_sequence_length - self.pool_length)/self.pool_layers*(self.pool_layers-layer-1), tf.int32), 
                               num_nodes)  # B, 单层，需要保留多少节点
            cluster_score = cluster_score * self.float_mask # B*L
            if 'kwai' in socket.gethostname():
                sorted_score = tf.contrib.framework.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_index = tf.stack([tf.range(tf.shape(Xc)[0]), tf.cast(to_keep, tf.int32)], 1) # B*2
                target_score = tf.gather_nd(sorted_score, target_index) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B)
            else:
                sorted_score = tf.sort(cluster_score, direction='DESCENDING', axis=-1) # B*L
                target_score = tf.gather_nd(sorted_score, tf.expand_dims(tf.cast(to_keep, tf.int32), -1), batch_dims=1) + K.epsilon() # indices[:-1]=(B) + data[indices[-1]=() --> (B) # 最终score
            topk_mask = tf.greater(cluster_score, tf.expand_dims(target_score, -1)) # B*L + B*1 -> B*L
            self.mask = tf.cast(topk_mask, tf.int32)
            self.float_mask = tf.cast(self.mask, tf.float32)
            self.reduced_sequence_length = tf.reduce_sum(self.mask, 1)

            ## ensure graph connectivity 
            E = E * tf.expand_dims(self.float_mask, -1) * tf.expand_dims(self.float_mask, -2)
            A = tf.matmul(tf.matmul(E, A_bool),
                          tf.transpose(E, (0,2,1))) # B*C*L x B*L*L x B*L*C = B*C*C
            ## graph readout 
            graph_readout = tf.reduce_sum(Xc*tf.expand_dims(cluster_score,-1)*tf.expand_dims(self.float_mask, -1), 1) # (B, F)

        return Xc, A, graph_readout, cluster_score
