import os
import numpy as np
import tensorflow as tf
from collections import deque

np_rnd = np.random
tf_conv = tf.contrib.layers.convolution2d
tf_dense = tf.layers.dense

_model_path = "_model/" 
_n_steps = 100000          
_train_start = 1000  
_train_interval = 3     # net training interval 
_save_steps = 50        # model saving interval
_copy_steps = 25        # weight copying interval
_skip_start = 90        # skip game start steps 
_batch_size = 50 
_discount_rate = 0.95   # for calculating target q value 

class DQNet:
    '''
    reference: https://arxiv.org/abs/1312.5602
    there, author shows a 4 layers net, including:
    1st: conv-16 kernel=(8, 8) stride=4 activation_fn=True
    2nd: conv-32 kernel=(4, 4) stride=2 activation_fn=True
    3rd: fully_connect-256 linear layer
    4th: output - fully_connect linear layer with {actions of game} units

    Here, based on: https://book.douban.com/subject/26840215/
    our model consisted by 5 layers, including:
    1st: conv-32 kernel=(8, 8) stride=4 activation_fn=tf.nn.relu
    2nd: conv-64 kernel=(4, 4) stride=2 activation_fn=tf.nn.relu
    3rd: conv-64 kernel=(3, 3) stride=1 activation_fn=tf.nn.relu 
    4th: fully_connect-512 activation_fn=tf.nn.relu
    5th: output - fully_connect linear layer with {actions of game} units
    '''
    def __init__(self, state_shape, opt_units,
                 action_shape=[None, 1], 
                 target_shape=[None, 1],
                 learning_rate = 0.01,
                 max_mem = 10000):

        self.replay_memory = deque([], maxlen=max_mem)
        self.opt_units = opt_units

        self.state = tf.placeholder(tf.float32, shape=state_shape)
        self.action = tf.placeholder(tf.int32, shape=action_shape)
        self.target = tf.placeholder(tf.float32, shape=target_shape)

        self.build_graph(learning_rate)
    
    def network(self, state, scope_name,
                conv_units = [32, 64, 64],
                kernels = [(8), (4), (3)],
                strides = [4, 2, 1],
                padding = 'SAME',
                conv_activation = tf.nn.relu,
                h_ipts = (64 * 11 * 10),
                h_units = 512,
                h_activation = tf.nn.relu,
                init_w = None):
        if init_w is None:
            init_w = tf.contrib.layers.variance_scaling_initializer()
        conv_layer = state
        with tf.variable_scope(scope_name) as scope:
            for u, k, s in zip(conv_units, kernels, strides):
                conv_layer = tf_conv(conv_layer,
                                     num_outputs = u, 
                                     kernel_size = k, 
                                     stride = s, 
                                     padding = padding,
                                     activation_fn = conv_activation,
                                     weights_initializer = init_w)
            h_input = tf.reshape(conv_layer, [-1, h_ipts])
            # hidden = tf_dense(h_input, h_units,
            #                   activation_fn = h_activation,
            #                   weights_initializer = init_w)
            # opt_q_val = tf_dense(hidden, self.opt_units,
            #                      activation_fn = None,
            #                      weights_initializer = init_w)
            hidden = tf_dense(h_input, h_units,
                              activation = h_activation,
                              kernel_initializer = init_w)
            opt_q_val = tf_dense(hidden, self.opt_units,
                                 activation = None,
                                 kernel_initializer = init_w)
        '''
        trainable_vars_by_name: collect all trainable variables 
        for copying to target actor network from training network
        it's a goal for divide target actor weight and training weight, detail in `Nature DQN`
        '''
        trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope=scope.name)                
        trainable_vars_by_name = {var.name[len(scope.name):]: var for var in trainable_vars}    
        return opt_q_val, trainable_vars_by_name

    def build_graph(self, learning_rate):
        actor_q_val, actor_vars = self.network(self.state, scope_name="q_networks/actor")
        train_q_val, train_vars = self.network(self.state, scope_name="q_networks/train")
        
        copy_ops = [actor_var.assign(train_vars[name]) for name, actor_var in actor_vars.items()]

        train_q_val = tf.reduce_sum(train_q_val * tf.one_hot(self.action, self.opt_units), 
                                    axis=1, keep_dims=True)
        loss = tf.reduce_mean(tf.square(self.target - train_q_val))
        optimizer = tf.train.AdamOptimizer(learning_rate)  
        train_op = optimizer.minimize(loss)

        self.copy_train_to_actor = tf.group(*copy_ops)  # package all copy operations into one op
        self.actor_q_val = actor_q_val 
        self.train_q_val = train_q_val 
        self.loss = loss 
        self.train_op = train_op

    def get_memory(self, batch_size):
        indices = np_rnd.permutation(len(self.replay_memory))[:batch_size]
        cols = [[], [], [], [], []] # state, action, reward, next_state, continues
        for idx in indices:
            mem = self.replay_memory[idx]
            for col, value in zip(cols, mem):            
                col.append(value)    
        cols = [np.array(col) for col in cols]
        return (cols[0], cols[1].reshape(-1, 1), cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1)) 

    def greedy(self, q_values, step, emin = 0.05, emax = 1.0, decay_steps = 50000):
        epsilon = max(emin, emax - (emax-emin) * step/decay_steps)   
        if np_rnd.rand() < epsilon:        
            return np_rnd.randint(self.opt_units)   
        else:        
            return np.argmax(q_values) 

    def init_sess_config(self, allow_growth=True, gpu_memory_frac=1.0):
        '''
            allow_growth: if True, gpu memory will increase by requirement 
            gpu_memory_frac: percentage of gpu memory use
        '''
        options = tf.GPUOptions(allow_growth=allow_growth,  
                                per_process_gpu_memory_fraction=gpu_memory_frac)
        return tf.ConfigProto(gpu_options=options)

    def training(self, env, game_name, preprocess_obs, 
              model_path = _model_path,
              start_steps = _skip_start,
              train_start = _train_start,
              train_interval = _train_interval,
              discount = _discount_rate,
              batch_size = _batch_size,
              sess_conf = None):
        train_step = 0
        iteration = 0
        done = True
        if sess_conf is None:
            sess_conf = self.init_sess_config()
        with tf.Session(config=sess_conf) as sess:
            saver = tf.train.Saver() 
            sess.run(tf.global_variables_initializer())
            checkpoint = tf.train.get_checkpoint_state(model_path)
            if checkpoint and checkpoint.model_checkpoint_path:                 
                saver.restore(sess, checkpoint.model_checkpoint_path)
                print ("Successfully loaded:", checkpoint.model_checkpoint_path)
            
            while True:
                if train_step >= _n_steps:
                    break
                iteration += 1
                if done: # restart game            
                    obs = env.reset()            
                    for _ in range(start_steps): # skip game start steps              
                        # env step return obs, reward, done, info
                        obs, reward, done, info = env.step(0)            
                    state = preprocess_obs(obs)

                env.render() 
                # get actor's q_values and action after last state        
                q_values = self.actor_q_val.eval(feed_dict={self.state: [state]})        
                action = self.greedy(q_values, train_step)

                # actor start play game       
                obs, reward, done, _ = env.step(action)        
                next_state = preprocess_obs(obs)

                # put (state, action, reward, next_state) into replay_memory        
                self.replay_memory.append((state, action, reward, next_state, 1.0 - done))        
                state = next_state
                if iteration >= train_start and iteration % train_interval == 0:                
                    tr_state, tr_action, tr_rewards, tr_next_state, continues = (self.get_memory(batch_size))        
                    next_q_val = self.actor_q_val.eval(feed_dict={self.state: tr_next_state})    
                    max_next_q_val = np.max(next_q_val, axis=1, keepdims=True)        
                    target_val = tr_rewards + continues * discount * max_next_q_val      
                    loss, _ = sess.run(
                                [self.loss, self.train_op],
                                feed_dict={
                                    self.state: tr_state, 
                                    self.action: tr_action, 
                                    self.target: target_val
                                }
                    )
                    train_step += 1
                    # print("Training in step - {}, Loss - {}".format(train_step, loss))

                if (train_step % _copy_steps) and (train_step != 0):            
                    self.copy_train_to_actor.run()

                if (train_step % _save_steps == 0) and (train_step != 0):            
                    saver.save(sess, (model_path + game_name))
                    print("Save model in step - {}".format(train_step))