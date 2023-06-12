#coding=utf-8
#pylint: disable=no-member
#pylint: disable=no-name-in-module
#pylint: disable=import-error
import warnings
warnings.filterwarnings("ignore")
from absl import app
from absl import flags
from absl import logging

# import sys
# sys.path.append("/data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE")
import sys
import os
import socket
import setproctitle
import tensorflow as tf

import time
from utils.deeprec_utils import prepare_hparams

from model import SURGEModel
from utils.io.sequential_iterator import SequentialIterator

from utils.visdom_utils import VizManager
from tensorboardX import SummaryWriter

FLAGS = flags.FLAGS
flags.DEFINE_string('name', 'taobao-SURGE', 'Experiment name.')
flags.DEFINE_string('dataset', 'taobao_global', 'Dataset name.')
flags.DEFINE_integer('gpu_id', 1, 'GPU ID.')
flags.DEFINE_integer('val_num_ngs', 4, 'Number of negative instances with a positiver instance for validation.')
flags.DEFINE_integer('test_num_ngs', 99, 'Number of negative instances with a positive instance for testing.')
flags.DEFINE_integer('batch_size', 512, 'Batch size.')
flags.DEFINE_integer('port', 8023, 'Port for visdom.') # for epoch visual
flags.DEFINE_string('model', 'SURGE', 'Model name.')
flags.DEFINE_float('embed_l2', 1e-6, 'L2 regulation for embeddings.')
flags.DEFINE_float('layer_l2', 1e-6, 'L2 regulation for layers.')
flags.DEFINE_integer('contrastive_length_threshold', 5, 'Minimum sequence length value to apply contrastive loss.')
flags.DEFINE_integer('contrastive_recent_k', 3, 'Use the most recent k embeddings to compute short-term proxy.')

flags.DEFINE_boolean('amp_time_unit', True, 'Whether to amplify unit for time stamp.')
flags.DEFINE_boolean('only_test', False, 'Only test and do not train.')
flags.DEFINE_boolean('test_dropout', False, 'Whether to dropout during evaluation.')
flags.DEFINE_boolean('write_prediction_to_file', False, 'Whether to write prediction to file.')
flags.DEFINE_boolean('test_counterfactual', False, 'Whether to test with counterfactual data.')
flags.DEFINE_string('test_counterfactual_mode', 'shuffle', 'Mode for counterfactual evaluation, could be original, shuffle or recent.')
flags.DEFINE_integer('counterfactual_recent_k', 10, 'Use recent k interactions to predict the target item.')
flags.DEFINE_boolean('pretrain', False, 'Whether to use pretrain and finetune.')
#  flags.DEFINE_boolean('finetune', True, 'Whether to use pretrain and finetune.')
#  flags.DEFINE_string('finetune_path', '/data/changjianxin/ls-recommenders/saves/GCN/gat-uii_last_pretrain/pretrain/', 'Save path.')
flags.DEFINE_string('finetune_path', '', 'Save path.')
flags.DEFINE_boolean('vector_alpha', False, 'Whether to use vector alpha for long short term fusion.')
flags.DEFINE_boolean('manual_alpha', False, 'Whether to use predefined alpha for long short term fusion.')
flags.DEFINE_float('manual_alpha_value', 0.5, 'Predifined alpha value for long short term fusion.')
flags.DEFINE_boolean('interest_evolve', True, 'Whether to use a GRU to model interest evolution.')
flags.DEFINE_boolean('predict_long_short', True, 'Predict whether the next interaction is driven by long-term interest or short-term interest.')
flags.DEFINE_enum('single_part', 'no', ['no', 'long', 'short'], 'Whether to use only long, only short or both.')
flags.DEFINE_integer('is_clip_norm', 1, 'Whether to clip gradient norm.')
flags.DEFINE_boolean('use_complex_attention', True, 'Whether to use complex attention like DIN.')
flags.DEFINE_boolean('use_time4lstm', True, 'Whether to use Time4LSTMCell proposed by SLIREC.')
flags.DEFINE_integer('epochs', 100, 'Number of epochs.')
flags.DEFINE_integer('early_stop', 7, 'Patience for early stop.')
flags.DEFINE_integer('pretrain_epochs', 10, 'Number of pretrain epochs.')
flags.DEFINE_integer('finetune_epochs', 100, 'Number of finetune epochs.')
flags.DEFINE_string('data_path', "/data/home/hejiansu/KT/xiangwei/rec/SIGIR21-SURGE/my_data/taobao_global_small", 'Data file path.')

from utils.util import get_time
dir_path = os.path.split(os.path.realpath(__file__))[0] # 当前文件所在的目录

flags.DEFINE_string('save_path', f'{dir_path}/../my_save/{get_time()}', 'Save path.')
#  flags.DEFINE_string('save_path', '../../saves_step/', 'Save path.')

flags.DEFINE_integer('train_num_ngs', 4, 'Number of negative instances with a positive instance for training.')
flags.DEFINE_float('sample_rate', 1.0, 'Fraction of samples for training and testing.')
flags.DEFINE_float('attn_loss_weight', 0.001, 'Loss weight for supervised attention.')
flags.DEFINE_float('discrepancy_loss_weight', 0.01, 'Loss weight for discrepancy between long and short term user embedding.')
flags.DEFINE_float('contrastive_loss_weight', 0.1, 'Loss weight for contrastive of long and short intention.')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate.')
flags.DEFINE_integer('show_step', 50, 'Step for showing metrics.')
flags.DEFINE_string('visual_type', 'epoch', '') #  for epoch visual
flags.DEFINE_integer('visual_step', 50, 'Step for drawing metrics.')
flags.DEFINE_string('no_visual_host', 'kwai', '')
flags.DEFINE_boolean('enable_mail_service', False, 'Whether to e-mail yourself after each run.')
flags.DEFINE_string('yaml_file', f'{dir_path}/gcn.yaml', 'yaml config')
flags.DEFINE_integer('seed', None, 'random seed')

def get_model(flags_obj, model_path, summary_path, pretrain_path, finetune_path, user_vocab, item_vocab, cate_vocab, train_num_ngs, data_path):

    EPOCHS = flags_obj.epochs
    BATCH_SIZE = flags_obj.batch_size
    RANDOM_SEED = None  # Set None for non-deterministic result

    flags_obj.amp_time_unit = flags_obj.amp_time_unit if flags_obj.model == 'DANCE' else False

    if flags_obj.dataset in ['taobao_global', 'yelp_global', 'taobao_global_small']:
        pairwise_metrics = ['mean_mrr', 'ndcg@2;4;6', 'hit@2;4;6']
        weighted_metrics = ['wauc']
        max_seq_length = 50
    elif flags_obj.dataset in ['assist09', 'assist09_small']:
        pairwise_metrics = None
        weighted_metrics = None
        max_seq_length = 200

    input_creator = SequentialIterator
    # import ipdb; ipdb.set_trace()
    if flags_obj.model == 'SURGE':
        yaml_file = flags_obj.yaml_file
        # import ipdb; ipdb.set_trace()
        hparams = prepare_hparams(yaml_file, 
                                embed_l2=flags_obj.embed_l2, 
                                layer_l2=flags_obj.layer_l2, 
                                learning_rate=flags_obj.learning_rate, 
                                epochs=EPOCHS,
                                EARLY_STOP=flags_obj.early_stop,
                                batch_size=BATCH_SIZE,
                                show_step=flags_obj.show_step,
                                visual_step=flags_obj.visual_step,
                                visual_type=flags_obj.visual_type,
                                MODEL_DIR=model_path,
                                SUMMARIES_DIR=summary_path,
                                PRETRAIN_DIR=pretrain_path,
                                FINETUNE_DIR=finetune_path,
                                user_vocab=user_vocab,
                                item_vocab=item_vocab,
                                cate_vocab=cate_vocab,
                                need_sample=False if flags_obj.dataset in ['assist09', 'assist09_small'] else True,
                                train_num_ngs=train_num_ngs, # provides the number of negative instances for each positive instance for loss computation.
                                max_seq_length=max_seq_length, 
                                hidden_size=40,
                                train_dir=os.path.join(data_path, r'train_data'),
                                graph_dir=os.path.join(data_path, 'graphs'),
                                pairwise_metrics=pairwise_metrics,
                                weighted_metrics=weighted_metrics,
                                dataset=flags_obj.dataset
                    )
        # import ipdb; ipdb.set_trace()
        model = SURGEModel(hparams, input_creator, seed=RANDOM_SEED)
    # import ipdb; ipdb.set_trace()
    return model


def main(argv):

    flags_obj = FLAGS

    setproctitle.setproctitle('{}'.format(flags_obj.name))
    os.environ["CUDA_VISIBLE_DEVICES"] = str(flags_obj.gpu_id)

    print("System version: {}".format(sys.version))
    print("Tensorflow version: {}".format(tf.__version__))

    print('start experiment')
    data_path = os.path.join(flags_obj.data_path, flags_obj.dataset)
   
    # for test
    train_file = os.path.join(data_path, r'train_data')
    valid_file = os.path.join(data_path, r'valid_data')
    test_file = os.path.join(data_path, r'test_data')
    user_vocab = os.path.join(data_path, r'user_vocab.pkl')
    item_vocab = os.path.join(data_path, r'item_vocab.pkl')
    cate_vocab = os.path.join(data_path, r'category_vocab.pkl')
    # output_file = os.path.join(data_path, r'output.txt')

    train_num_ngs = flags_obj.train_num_ngs
    valid_num_ngs = flags_obj.val_num_ngs
    test_num_ngs = flags_obj.test_num_ngs

    # tensorboard 可视化
    save_path = os.path.join(flags_obj.save_path, flags_obj.model, flags_obj.name)
    model_path = os.path.join(save_path, "model/")
    summary_path = os.path.join(save_path, "summary/")
    pretrain_path = os.path.join(save_path, "pretrain/")
    # finetune_path = os.path.join(flags_obj.save_path, 'GCN/gat-uii_last_pretrain/' ,"pretrain/")
    finetune_path = flags_obj.finetune_path
    
    model = get_model(flags_obj, model_path, summary_path, pretrain_path, finetune_path, user_vocab, item_vocab, cate_vocab, train_num_ngs, data_path)
    # import ipdb; ipdb.set_trace()
    # vm 可视化
    if flags_obj.no_visual_host not in socket.gethostname():
        vm = VizManager(flags_obj)
        vm.show_basic_info(flags_obj)
    else:
        vm = None
    #  visual_path = summary_path
    visual_path = os.path.join(save_path, "metrics/")
    tb = SummaryWriter(log_dir=visual_path, comment='tb')

    #  print(model.run_weighted_eval(test_file, num_ngs=test_num_ngs)) # test_num_ngs is the number of negative lines after each positive line in your test_file

    if flags_obj.dataset in ['taobao_global', 'yelp_global', 'taobao_global_small']:
        eval_metric = 'wauc'
    elif flags_obj.dataset in ['assist09', 'assist09_small']:
        eval_metric = 'auc'
    

    start_time = time.time()
    # train
    model = model.fit(train_file, 
                      valid_file, 
                      valid_num_ngs=valid_num_ngs, 
                      eval_metric=eval_metric, 
                      vm=vm, 
                      tb=tb, 
                      pretrain=flags_obj.pretrain) 
    
    # valid_num_ngs is the number of negative lines after each positive line in your valid_file 
    # we will evaluate the performance of model on valid_file every epoch
    end_time = time.time()
    cost_time = end_time - start_time
    print('Time cost for training is {0:.2f} mins'.format((cost_time)/60.0))

    # test
    ckpt_path = tf.train.latest_checkpoint(model_path)
    model.load_model(ckpt_path)
    res_syn = model.run_weighted_eval(test_file, num_ngs=test_num_ngs)
    print(flags_obj.name)
    print(res_syn)

    # test basedon groub
    # for g in [1,2,3,4,5]:
    #     res_syn_group = model.run_weighted_eval(test_file+'_group'+str(g), num_ngs=test_num_ngs)
    #     print(flags_obj.name+'_group'+str(g))
    #     print(res_syn_group)
    
    if flags_obj.no_visual_host not in socket.gethostname():
        vm.show_test_info()
        vm.show_result(res_syn)

    tb.close()

if __name__ == "__main__":
    
    app.run(main)
