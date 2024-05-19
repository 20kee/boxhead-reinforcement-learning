
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

        self.dx = [-1, -1, 0, 1, 1, 1, 0, -1]
        self.dy = [0, 1, 1, 1, 0, -1, -1, -1]
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

        self.before_key = ''
        self.epsilon = 0.1
        self.true_count = 0
        self.false_count = 0
        self.before = False
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        self.REINFORCE_model = self.generate_model()
        
    
    def generate_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=128, input_dim=5))
        model.add(tf.keras.layers.Dense(units=64, input_dim=128, activation='relu'))
        model.add(tf.keras.layers.Dense(units=32, input_dim=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=8, input_dim=32, activation='softmax'))
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
    
    def bomb(self):
        pag.keyDown('right')
        time.sleep(0.02)
        pag.keyUp('right')
        time.sleep(0.02)

        pag.keyDown('Up')
        time.sleep(0.02)
        pag.keyUp('Up')
        time.sleep(0.02)

        pag.keyDown('space')
        time.sleep(0.02)
        pag.keyUp('space')
        time.sleep(0.02)

        pag.keyDown('right')
        time.sleep(0.02)
        pag.keyUp('right')
        time.sleep(0.02)

        pag.keyDown('down')
        time.sleep(0.02)
        pag.keyUp('down')
        time.sleep(0.02)

        pag.keyDown('space')
        time.sleep(0.02)
        pag.keyUp('space')
        time.sleep(0.02)


    def game_restart(self):
        self.remove_ad()
        pag.click(*self.restart_game)

    def get_min_dist(self, agent, jombie):
        dists = [
            self.get_dist(agent[0], agent[1], 0, agent[0], agent[1]-416, 0, jombie[0], jombie[1], 0),
            self.get_dist(agent[0], agent[1], 0, agent[0]+416, agent[1]-416, 0, jombie[0], jombie[1], 0),
            self.get_dist(agent[0], agent[1], 0, agent[0]+416, agent[1], 0, jombie[0], jombie[1], 0),
            self.get_dist(agent[0], agent[1], 0, agent[0]+416, agent[1]+416, 0, jombie[0], jombie[1], 0),
            
            self.get_dist(agent[0], agent[1], 0, agent[0], agent[1]+416, 0, jombie[0], jombie[1], 0),
            self.get_dist(agent[0], agent[1], 0, agent[0]-416, agent[1]+416, 0, jombie[0], jombie[1], 0),
            self.get_dist(agent[0], agent[1], 0, agent[0]-416, agent[1], 0, jombie[0], jombie[1], 0),
            self.get_dist(agent[0], agent[1], 0, agent[0]-416, agent[1]-416, 0, jombie[0], jombie[1], 0)
        ]
        min_dist = min(dists)

        return (min_dist, dists.index(min_dist))
    
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

    def generate_action(self, state):
        if state[1] >= 340:
            pag.keyDown('up')
            time.sleep(0.05)
            pag.keyUp('up')

        action_prob_tensor = np.array(self.REINFORCE_model(tf.convert_to_tensor([state]))[0])
        action = np.random.choice(len(action_prob_tensor), 1, p=action_prob_tensor)[0]
        return action, action_prob_tensor[action]
    
    def do_action(self, action, index):
        direction = (self.dx[index], self.dy[index])
        keys = ['' if direction[0] == 0 else ('up' if direction[0] == -1 else 'down'),
                    '' if direction[1] == 0 else ('left' if direction[1] == -1 else 'right')]
        
        pag.press(keys)
        time.sleep(0.02)

        pag.keyDown('space')
        time.sleep(0.02)
        pag.keyUp('space')

        direction = (self.dx[action], self.dy[action])
        keys = ['' if direction[0] == 0 else ('up' if direction[0] == -1 else 'down'),
                '' if direction[1] == 0 else ('left' if direction[1] == -1 else 'right')]
        
        pag.press(keys)
        time.sleep(0.1)
        
        print('do action', action)

    def generate_reward(self, state):
        return 1


    def generate_state(self):
        img = ImageGrab.grab(bbox=(*self.left_top, *self.right_bottom))
        img = img.resize((416, 416))
        results = self.model.predict(source=img)
            # state = self.generate_state(results[0].boxes.data)
        objects = results[0].boxes.data
        # 상태 = 가장 가까운 두 좀비의 상대적 위치 + 내 시선과 좀비와의 최소 거리
        agent = []
        jombies = []
        for object in objects:
            # 0 : agent, 1 : ball, 2 : boss, 3 : jombie
            object = list(map(float, object))
            cls = int(round(object[-1]))
            if cls == 0:
                agent.append( ( (object[0]+object[2])/2, (object[1]+object[3])/2 ) )
            elif cls != 1:
                jombies.append( ( (object[0]+object[2])/2, (object[1]+object[3])/2 ) )
        
        if len(agent) == 0 or len(jombies) == 0:
            return False, 0

        jombie_distance = []
        for i, jombie in enumerate(jombies):
            jombie_distance.append( (((jombie[0]-agent[0][0])**2 + (jombie[1]-agent[0][1])**2) ** (1/2), i) )
        jombie_distance.sort()

        state = []
        
        state.extend([agent[0][0]/416, agent[0][1]/416])
        state.extend([jombies[jombie_distance[0][1]][0]/416, jombies[jombie_distance[0][1]][1]/416])
        result = self.get_min_dist(agent[0], jombies[jombie_distance[0][1]])    
        state.append(result[0]/100)
        print(agent[0])
        return state, result[1]

    def make_episode(self):
        T = []
        state = False
        index = 0
        start = time.time()
        while True:
            state, index = self.generate_state()
            if state:
                self.bomb()
                break

            if time.time() - start >= 10:
                return T

        while True:
            action, prob = self.generate_action(state)
            self.do_action(action, index)

            while True:
                next_state, index = self.generate_state()
                if next_state:
                    break
                else:
                    self.false_count += 1
                    if self.false_count == 25:
                        self.false_count = 0

                        return T

            reward = self.generate_reward(next_state)
            T.append([state, prob, action, reward])

            state = next_state

    def train_model(self, T):
        states = []
        actions = []
        for t in T:
            states.append(t[0])
            act = [0, 0, 0, 0, 0, 0, 0, 0 ]
            act[t[2]] = 1
            actions.append(act)
        
        rewards = []
        G = 0    
        for t in T[::-1]:
            G = G * 0.99 + t[3]
            rewards.append(G)
        rewards = rewards[::-1]
        states = np.asarray(states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
    
        with tf.GradientTape() as tape:
            policies = self.REINFORCE_model(states)
            action_prob = tf.reduce_sum(actions * policies, axis=1)
            cross_entropy = -tf.math.log(action_prob + 1e-5)
            loss = tf.reduce_sum(cross_entropy * rewards)

        gradients = tape.gradient(loss, self.REINFORCE_model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.REINFORCE_model.trainable_variables))


    def train(self):
        i = 0
        while True:
            i += 1
            T = self.make_episode()
            if len(T) > 0:
                T[-1][3] = -3
                self.train_model(T)
            self.game_restart()

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
            

boxhead = Boxhead()
boxhead.main()