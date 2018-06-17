
import gym 
from model import DQNet

game_name = {
    'pong' : 'Pong-v0',
    'pacman' : 'MsPacman-v0'
}

def preprocess(obs, game=game_name['pacman']):
    if game == 'MsPacman-v0':
        img = obs[1:176:2, ::2]         # cut
    elif game == 'Pong-v0':
        img = obs[25:200:2, ::2]         # cut
    img = img.mean(axis=2)          # gray
    img = (img - 128) / 128 - 1     # regularization   
    return img.reshape(88, 80, 1) 

def main():
    
    env = gym.make(game_name['pong'])
    net = DQNet(state_shape=(None, 88, 80, 1), opt_units=env.action_space.n)
    net.training(env, game_name['pong'], preprocess)

if __name__ == '__main__':
    main()