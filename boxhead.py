
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
import pyautogui as pag
import pyscreenshot as ImageGrab
import time
import base64
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import platform
import keyboard
from ultralytics import YOLO
import threading

class Boxhead:
    def __init__(self):
        self.boxhead_url = 'https://flgamenara.tistory.com/6#google_vignette'
        self.driver = ''
        self.action = ''
        self.temp = True
        self.dx = [-1, -1, -1, 0, 0, 1, 1, 1]
        self.dy = [-1 ,0, 1, -1, 1, -1, 0, 1]
        self.orientation = [0, 1]
        self.model = YOLO('/Users/20kee/Desktop/programming/Language/python/boxhead-reinforcement-learning/runs/detect/train/weights/best.pt')
        if platform.system() == "Windows":
            print('window')
            self.left_top = (24, 100)
            self.right_bottom = (738, 560)
            self.start = (350, 250)
            self.single_play = (393, 354)
            self.next_map = (646, 363)
            self.start_game = (443, 359)
            self.restart_game = (381, 484)
        else:
            print('Mac')
            self.left_top = (80, 160)
            self.right_bottom = (800, 700)
            self.start = (380, 400)
            self.single_play = (393, 400)
            self.next_map = (692, 400)
            self.start_game = (465, 370)
            self.restart_game = (415, 520)

        self.epsilon = 0.5
        self.true_count = 0
        self.false_count = 0
        self.before = False
        self.sars_pool = []
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.DQN_model = self.generate_model()
        self.DQN_target_model = self.generate_model()
        
    
    def generate_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=64, input_dim=5))
        model.add(tf.keras.layers.Dense(units=48, input_dim=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=32, input_dim=48, activation='relu'))
        model.add(tf.keras.layers.Dense(units=8, input_dim=32,
                            kernel_initializer=tf.keras.initializers.RandomUniform(-1e-3, 1e-3)))
        model.compile(optimizer = self.optimizer)
        return model
    
    def remove_ad(self):
        try:
            iframe = self.driver.find_element(By.ID, 'aswift_6')
            self.driver.switch_to.frame(iframe)
            try:
                self.driver.find_element(By.ID, 'dismiss-button').click()
            except:
                pass
            self.driver.switch_to.parent_frame()
        except:
            pass

    def game_start(self):
        pag.click(*self.start)
        time.sleep(21)
        self.remove_ad()
        pag.click(*self.single_play)
        time.sleep(1)

        for _ in range(10):
            self.remove_ad()
            pag.click(*self.next_map)
            time.sleep(0.2)

        pag.click(*self.start_game)

    def game_restart(self):
        self.remove_ad()
        pag.click(*self.restart_game)

    def get_dist(self, ax, ay, az, bx, by, bz, cx, cy, cz):
        def calculate(x1, y1, z1, x2, y2, z2):                              # 두 점 사이의 거리 계산
            return ((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2) ** (1/2)
        
        minlen = 1e9
        lenSC = calculate(ax, ay, az, cx, cy, cz)
        lenEC = calculate(bx, by, bz, cx, cy, cz)

        while True:
            mx = (ax + bx) / 2
            my = (ay + by) / 2
            mz = (az + bz) / 2
            lenKC = calculate(mx, my, mz, cx, cy, cz)           # mid(임의의 점)과 C 사이의 거리
            if abs(minlen - lenKC) <= 1e-6:
                return lenKC
            minlen = min(lenKC, minlen)
            if lenSC < lenEC:              						# [시작점과 C 사이의 거리]가 [끝점과 C 사이의 거리]보다 작을때, 끝점을 시작점쪽으로 당기기
                bx, by, bz = mx, my, mz
                lenEC = lenKC
            else:
                ax, ay, az = mx, my, mz
                lenSC = lenKC

    def generate_action(self, state, epsilon=0.1):
        epsilon = self.epsilon
        if np.random.rand() <= epsilon:
            return random.randrange(8)
        else:
            q_value = self.DQN_model(np.array([state]))
            return np.argmax(q_value[0])
        
    def do_action(self, action):
        print('action', action)
        
        self.orientation = [self.dx[action], self.dy[action]]
        self.leftright = '' if self.orientation[1] == 0 else ('left' if self.orientation[1] == -1 else 'right')
        self.updown = '' if self.orientation[0] == 0 else ('up' if self.orientation[1] == -1 else 'down')
        print(self.leftright, self.updown)
        pag.keyDown(self.leftright)
        pag.keyDown(self.updown)
        time.sleep(0.35)
        pag.keyUp(self.leftright)
        pag.keyUp(self.updown)

    def generate_reward(self, state):
        if state[2] != 0:
            minus_reward = 2*state[4]
        else:
            minus_reward = state[4]
        reward = (state[0]**2 + state[1]**2)**(1/2) + (state[2]**2 + state[3]**2)**(1/2) - minus_reward
        print('reward', reward)
        return reward


    def generate_state(self, objects):
        # 상태 = 가장 가까운 두 좀비의 상대적 위치 + 내 시선과 좀비와의 최소 거리
        agent = []
        jombies = []
        jombie_distance = []
        for object in objects:
            # 0 : agent, 1 : ball, 2 : boss, 3 : jombie
            object = list(map(float, object))
            cls = int(round(object[-1]))
            if cls == 0:
                agent.append( ( (object[0]+object[2])/2, (object[1]+object[3])/2 ) )
            elif cls != 1:
                jombies.append( ( (object[0]+object[2])/2, (object[1]+object[3])/2 ) )
        
        if len(agent) == 0 and len(jombies) == 0:
            self.false_count += 1
            if self.false_count >= 15:
                self.false_count = 0
                return 'replay'

        if len(agent) == 0 or len(jombies) == 0: 
            return False
        agent = agent[0]
        
        # state = 가장 가까운 좀비 상대위치 + 두번째로 가까운 좀비 상대위치 + 내가 보는 방향으로 뻗은 직선과 가장 가까운 좀비와의 거리
        for i, jombie in enumerate(jombies):
            jombie_distance.append( (((jombie[0]-agent[0])**2 + (jombie[1]-agent[1])**2) ** (1/2), i) )
        jombie_distance.sort()

        state = []
        if len(jombie_distance) == 1:
            jombie1 = jombies[jombie_distance[0][1]]
            state.extend([jombie1[0]-agent[0], jombie1[1]-agent[1], 0, 0])
        else:
            jombie2 = jombies[jombie_distance[1][1]]
            jombie1 = jombies[jombie_distance[0][1]]
            state.extend([jombie1[0]-agent[0], jombie1[1]-agent[1],
                          jombie2[0]-agent[0], jombie2[1]-agent[1]])
        
        jombie3 = jombies[jombie_distance[0][1]]
        dist = self.get_dist(agent[0], agent[1], 0, agent[0]+416*self.orientation[0], agent[1]+416*self.orientation[1], 0, jombie3[0], jombie3[1], 0)
        state.append(dist)
        return state
    
    def train_model(self):
        train_pool = random.sample(self.sars_pool, 30)
        states = np.array([sars[0] for sars in train_pool])
        actions = np.array([sars[1] for sars in train_pool])
        rewards = np.array([sars[2] for sars in train_pool])
        next_states = np.array([sars[3] for sars in train_pool])

        with tf.GradientTape() as tape:
            outputs = self.DQN_model(states)
            one_hot_actions = tf.one_hot(actions, 2)
            predicts = tf.reduce_sum(one_hot_actions * outputs, axis=1)

            target_predicts = self.DQN_target_model(next_states)
            target_predicts = tf.stop_gradient(target_predicts)

            max_q = np.amax(target_predicts, axis=-1)
            targets = rewards + 0.995 * max_q
            loss = tf.reduce_mean(tf.square(targets - predicts))
        
        gradients = tape.gradient(loss, self.DQN_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.DQN_model.trainable_variables))
    

    def shoot(self):
        while True:
            pag.hotkey('space')

    def train(self):
        t = threading.Thread(target=self.shoot)
        t.start()

        i = 0
        before_state = []
        while True:
            i += 1
            img = ImageGrab.grab(bbox=(*self.left_top, *self.right_bottom))
            img = img.resize((416, 416))
            results = self.model.predict(source=img, save=True)
            state = self.generate_state(results[0].boxes.data)
            if state == False:
                pass
            elif state == 'replay':
                before_state = []
                self.game_restart()
            else:
                if before_state != []:
                    reward = self.generate_reward(state)
                    self.sars_pool.append([before_state, action, reward, state])
                    if len(self.sars_pool) >= 150:
                        print("start training")
                        time.sleep(5)
                        self.train_model()
                        self.DQN_target_model.set_weights(self.DQN_model.get_weights())
                        self.epsilon *= 0.95
            
                action = self.generate_action(state)
                self.do_action(action)
        

    def main(self):
        options = webdriver.ChromeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.action = ActionChains(self.driver)
        self.driver.get(self.boxhead_url)

        while True:
            try:
                self.ruffle = self.driver.find_element(By.TAG_NAME, 'ruffle-embed')
                self.driver.execute_script("window.scrollTo({}, {})".format(self.ruffle.location['x'], self.ruffle.location['y']))
                print("good")
                break
            except Exception as e:
                print(e)
                pass
        
        time.sleep(2)
        while True:
            self.remove_ad()
            self.game_start()
            self.train()
            
                


        time.sleep(10000)

boxhead = Boxhead()
# def key_down(e):
#     if e.name == 'command':
#         boxhead.send_key()


# keyboard.hook(key_down)
boxhead.main()