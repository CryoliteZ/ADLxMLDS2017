from agent_dir.agent import Agent
import scipy
import numpy as np
import gym
from keras.models import Sequential
from keras.layers import Dense, Reshape, Flatten, Input
from keras.optimizers import Adam
from keras.layers.convolutional import Convolution2D
import os, csv
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def prepro(o,image_size=[80,80]):
    """
    Call this function to preprocess RGB image to grayscale image if necessary
    This preprocessing code is from
        https://github.com/hiwonjoon/tf-a3c-gpu/blob/master/async_agent.py
    
    Input: 
    RGB image: np.array
        RGB screen of game, shape: (210, 160, 3)
    Default return: np.array 
        Grayscale image, shape: (80, 80, 1)
    
    """
    y = 0.2126 * o[:, :, 0] + 0.7152 * o[:, :, 1] + 0.0722 * o[:, :, 2]
    y = y.astype(np.uint8)
    resized = scipy.misc.imresize(y, image_size)
    return np.expand_dims(resized.astype(np.float32),axis=2)


class Agent_PG(Agent):
    def __init__(self, env, args):
        """
        Initialize every things you need here.
        For example: building your model
        """

        super(Agent_PG,self).__init__(env)
        self.env = env
        self.state_size = (80, 80, 1)
        self.action_size = 3
        self.gamma = 0.95
        self.learning_rate = 0.001
        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []
        self.model = self.build_model()
        self.model.summary()
        self.prev_x = None
        self.model_id = 0
        self.scores = []
        self.best_score = -22
        if args.test_pg:
            #you can load your model here
            print('loading trained model')
            self.load(os.path.join("model_best_pong", "pong_best.h5"))


    def init_game_setting(self):
        """

        Testing function will call this function at the begining of new game
        Put anything you want to initialize if necessary

        """
        # game actions
        # 0 1 no move
        # 2 4 up
        # 3 5 down
        pass

    def build_model(self):
        model = Sequential()
        # model.add(Reshape((80, 80, 1), input_shape=(self.state_size)))
        model.add(Convolution2D(32, kernel_size = (6, 6), strides=(3, 3), border_mode='same',
                                activation='relu', init='he_uniform', input_shape=(80, 80, 1)))
        model.add(Flatten())
        model.add(Dense(64, activation='relu', init='he_uniform'))
        model.add(Dense(32, activation='relu', init='he_uniform'))
        model.add(Dense(self.action_size, activation='softmax'))
        opt = Adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        return model

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(np.array(y).astype('float32') - prob)
        self.states.append(state)
        self.rewards.append(reward)

    # def act(self, state):
    #     # state = state.reshape([1, state.shape[0], state.shape[1], state.shape[2]])
    #     state = state.reshape([1, state.shape[0]])
    #     aprob = self.model.predict(state, batch_size=1).flatten()
    #     self.probs.append(aprob)
    #     prob = aprob / np.sum(aprob)
    #     action = np.random.choice(self.action_size, 1, p=prob)[0]
    #     return action, prob

    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, rewards.size)):
            if rewards[t] != 0:
                running_add = 0
            running_add = running_add * self.gamma + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    def train_on_batch(self):
        gradients = np.vstack(self.gradients)
        rewards = np.vstack(self.rewards)
        rewards = self.discount_rewards(rewards)
        rewards =  (rewards - np.mean(rewards)) / np.std(rewards)
        gradients *= rewards
        X = np.squeeze(np.vstack([self.states]), axis = 1)
        Y = self.probs + self.learning_rate * np.squeeze(np.vstack([gradients]))
        self.model.train_on_batch(X, Y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
    
    
    def preprocess(self,I):
        I = I[35:195]
        I = I[::2, ::2, 0]
        I[I == 144] = 0
        I[I == 109] = 0
        I[I != 0] = 1
        I = I.astype(np.float).ravel()
        return I.reshape((80,80,1))


    def train(self):
        """
        Implement your training algorithm here
        """

        if not (os.path.exists("model/%d/" % self.model_id)):
            os.makedirs("model/%d/" % self.model_id)
        
        with open(os.path.join("model", str(self.model_id) , "episodes"), "r") as f:
            if(f):
                # episode = len(f.readlines())
                rows = csv.reader(f, delimiter=',')
                episode = 0
                for row in rows:
                    episode += 1
                    self.scores.append(float(row[1]))
            else:
                episode = 0
        if(episode > 0):
            self.load(os.path.join("model", str(self.model_id) , "pong.h5"))
        
        state = self.env.reset()
        score = 0
        
        while True:       

            action, prob, x = self.make_action(state, test=False)
            state, reward, done, info = self.env.step(action+1)
            score += reward          
            self.remember(x, action, prob, reward)

            if done:
                episode += 1
                # record stuff
                self.scores.append(score)      
                moving_window = min(len(self.scores), 30)
                moving_avg = 0
                for i in range(episode - moving_window, episode):
                    moving_avg += self.scores[i]
                moving_avg /= moving_window

                print('Episode: %d - Score: %f. , moving %f' % (episode, score, moving_avg))
                
                model_path = os.path.join("model", str(self.model_id) , "pong.h5")
                best_model_path = os.path.join("model", str(self.model_id) , "pong_best.h5")
                with open(os.path.join("model", str(self.model_id) , "episodes"), "a") as f:
                    f.write(str(episode) + "," + str(score) + "," + str(moving_avg) + "\n")
                
                self.train_on_batch()
                score = 0
                state = self.env.reset()
                self.prev_x = None
                
                if moving_avg > self.best_score:
                    self.save(best_model_path)
                    self.best_score = moving_avg
                if episode > 1 and episode % 5 == 0:
                    self.save(model_path)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
        
            


    def make_action(self, observation, test=True):
        """
        Return predicted action of your agent

        Input:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)

        Return:
            action: int
                the predicted action from trained model
        """

        state = self.preprocess(observation)
        x = state - self.prev_x if self.prev_x is not None else np.zeros(self.state_size)
        self.prev_x = state

        x = np.expand_dims(x,axis=0)

        aprob = self.model.predict(x, batch_size=1).flatten()
        self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]

        if(test):
            return action + 1
        else:
            return action, prob, x
       

