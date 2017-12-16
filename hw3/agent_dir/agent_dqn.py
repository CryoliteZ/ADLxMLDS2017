from agent_dir.agent import Agent
import tensorflow as tf 
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Lambda, Dense, Activation, GRU, TimeDistributed, Conv1D, MaxPooling1D, MaxPooling2D, Flatten, Dropout, SimpleRNN, Bidirectional, Convolution2D, Permute, BatchNormalization, RepeatVector, Input, MaxPool2D 
from keras.models import load_model, Model
from keras.layers import merge, Concatenate, Merge, add, Masking, Activation, dot, multiply, concatenate
from keras import backend as K
import random, time, os, csv
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

class Agent_DQN(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """
        super(Agent_DQN,self).__init__(env)
        self.env = env
        self.state = env.reset()
        self.action_size = env.action_space
        self.memory = deque(maxlen=50000)
        self.gamma = 0.99    
        self.epsilon = 0.001
        self.epsilon_min = 0.0001
        self.epsilon_decay = 0.995
        self.learning_rate = 0.002
        self.graph_ops = None
        self.session = None
        self.scores = []

        if args.test_dqn:
            #you can load your model here
            print('loading trained model')

        

    def build_network(self, num_actions, agent_history_length, resized_width, resized_height):
        state = tf.placeholder("float", [None, resized_width, resized_height, agent_history_length])
        # print(resized_width, resized_height, agent_history_length, num_actions)
        inputs = Input(shape=(resized_width, resized_height,agent_history_length))
        model = Convolution2D(32, (3,3), activation='relu', padding='same')(inputs)
        model = MaxPool2D((2,2))(model)
        model = Convolution2D(64, (3,3), activation='relu', padding='same')(model)
        model = MaxPool2D((2,2))(model)
        model = Flatten()(model)
        model = Dense(512, activation='relu')(model)
        q_values = Dense(num_actions, activation='linear')(model)
        model = Model(inputs, q_values)
        return state, model

    def build_graph(self, num_actions):
        #  online Q-network
        s, q_network = self.build_network(num_actions, 4, 84, 84)
        network_params = q_network.trainable_weights
        q_values = q_network(s)
        #  target Q-network
        st, target_q_network = self.build_network(num_actions, 4, 84, 84)
        target_network_params = target_q_network.trainable_weights
        target_q_values = target_q_network(st)
        # update target network with online network weights
        update_target_network = [target_network_params[i].assign(network_params[i]) for i in range(len(target_network_params))]
        
        # cost and gradient
        a = tf.placeholder("float", [None, num_actions])
        y = tf.placeholder("float", [None])
        action_q_values = tf.reduce_sum(q_values * a, reduction_indices=1)
        cost = tf.reduce_mean(tf.square(y - action_q_values))
        optimizer = tf.train.AdamOptimizer(0.001)
        update_online_network = optimizer.minimize(cost, var_list=network_params)

        graph_ops = {"s" : s, 
                     "q_values" : q_values,
                     "st" : st, 
                     "target_q_values" : target_q_values,
                     "update_target_network" : update_target_network,
                     "a" : a,
                     "y" : y,
                     "update_online_network" : update_online_network}
        return graph_ops

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def init_game_setting(self):
        # to prevent endless games
        self.epsilon = 0.001
        g = tf.Graph()
        self.session = tf.Session(graph=g)
        with g.as_default(), self.session.as_default():
            K.set_session(self.session)
            self.graph_ops = self.build_graph(3)
            saver = tf.train.Saver()
            saver.restore(self.session, "model_best_dqn/model.ckpt")
            self.session.run(self.graph_ops["update_target_network"])
       
    def replay(self, batch_size, session, graph_ops):
        # sample minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        X_state, X_action, Y = [], [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * \
                       np.amax(graph_ops['target_q_values'].eval(session = session, feed_dict = {graph_ops['st'] : [next_state]}))

            action_input = np.zeros(3)
            action_input[action-1] = 1
            Y.append(target)
            X_state.append(state)
            X_action.append(action_input)

        session.run(graph_ops['update_online_network'], feed_dict = { graph_ops['y'] : np.asarray(Y),
                                                            graph_ops['a'] : np.asarray(X_action),
                                                            graph_ops['s'] : np.asarray(X_state)} )
        # epsilon decay                                    
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


    def train(self):
        """
        Implement your training algorithm here
        """
        if not (os.path.exists("model_dqn/%d/" % self.model_id)):
            os.makedirs("model_dqn/%d/" % self.model_id)
        with open(os.path.join("model_dqn", str(self.model_id) , "episodes"), "a") as f:
            f.close()
        episode = 0
        with open(os.path.join("model_dqn", str(self.model_id) , "episodes"), "r") as f:
            if(f):
                # episode = len(f.readlines())
                rows = csv.reader(f, delimiter=',')
                episode = 0
                for row in rows:
                    episode += 1
                    # self.scores.append(float(row[1]))
                    self.epsilon = float(row[3])
            else:
                episode = 0
            
        model_path = os.path.join("model_dqn", str(self.model_id) , "model.ckpt")
        g = tf.Graph()
        self.session = tf.Session(graph=g)
        with g.as_default(), self.session.as_default():
            K.set_session(self.session)
            graph_ops = self.build_graph(3)
            saver = tf.train.Saver()
            # for model reload 
            if(episode == 0):
                self.session.run(tf.global_variables_initializer())
                self.session.run(graph_ops["update_target_network"])
            else:
                saver.restore(self.session, model_path)
                self.session.run(graph_ops["update_target_network"])
            
            max_episodes = 100000
            max_timestep = 10000
            self.graph_ops = graph_ops
            for e in range(episode, max_episodes):
                # reset state
                env = self.env
                state = env.reset()
                total_reward = 0
                for time_t in range(max_timestep):
                    # ditermine action
                    action = self.make_action(state)
                    # Next step
                    next_state, reward, done, info = self.env.step(action)
                    total_reward += reward
                    # Remember  state, action and  reward
                    self.remember(state, action, reward, next_state, done)
                    
                    # make next_state the new current state for the next frame.
                    state = next_state

                    if done:
                        # record scores
                        if(len(self.scores) == 30):
                            del self.scores[0]
                            self.scores.append(total_reward) 
                        else:
                            self.scores.append(total_reward)
                        moving_avg = sum(self.scores) / len(self.scores)

                        print("episode: {}/{}, score: {}, moving_avg: {:.4f}, epsilon: {}, "
                              .format(e, max_episodes, total_reward, moving_avg, self.epsilon ))

                        with open(os.path.join("model_dqn", str(self.model_id) , "episodes"), "a") as f:
                            f.write(str(e) + "," + str(total_reward) + "," + str(moving_avg) + "," + str(self.epsilon)+ "\n")
                        break
                    
                # minibatch update and train online netwrok
                replay = len(self.memory) if len(self.memory) < 64 else 64
                self.replay(replay, self.session, graph_ops)

                # copy Q-netwark to update target network
                if e % 100 == 0:
                    self.session.run(graph_ops['update_target_network'])

                # save model
                if e % 100 == 0:
                    saver.save(self.session, model_path)


    def make_action(self, observation, test=True):
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)+1
        action = self.graph_ops['q_values'].eval(session = self.session, feed_dict = {self.graph_ops['s'] : [observation]})
        return np.argmax(action)+1
        

