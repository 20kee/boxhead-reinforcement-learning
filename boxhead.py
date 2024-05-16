
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
import matplotlib.pyplot as plt
import tensorflow as tf
import platform
import keyboard
from ultralytics import YOLO

class Boxhead:
    def __init__(self):
        self.boxhead_url = 'https://flgamenara.tistory.com/6#google_vignette'
        self.driver = ''
        self.action = ''
        self.temp = True
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
            print(1)
            self.left_top = (80, 160)
            self.right_bottom = (800, 700)
            self.start = (380, 400)
            self.single_play = (393, 400)
            self.next_map = (692, 400)
            self.start_game = (465, 370)
            self.restart_game = (395, 505)

        self.sars_pool = []
        self.DQN_model = self.generate_model()
        self.DQN_target_model = self.generate_model()
    
    def generate_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(units=64, input_dim=5))
        model.add(tf.keras.layers.Dense(units=48, input_dim=64, activation='relu'))
        model.add(tf.keras.layers.Dense(units=32, input_dim=48, activation='relu'))
        model.add(tf.keras.layers.Dense(units=8, input_dim=32))
        model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.01))
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
            state.extend([jombie1[0]-agent[0], jombie1[1]-agent[1], agent[0], agent[1]])
        else:
            jombie2 = jombies[jombie_distance[1][1]]
            jombie1 = jombies[jombie_distance[0][1]]
            state.extend([jombie1[0]-agent[0], jombie1[1]-agent[1],
                          jombie2[0]-agent[0], jombie2[1]-agent[1]])
        

        jombie3 = jombies[jombie_distance[0][1]]
        dist = self.get_dist(agent[0], agent[1], 0, agent[0]+416*self.orientation[0], agent[1]+416*self.orientation[1], 0, jombie3[0], jombie3[1], 0)
        state.append(dist)
        return state

    def get_image(self):
        i = 0
        while True:
            i += 1
            img = ImageGrab.grab(bbox=(*self.left_top, *self.right_bottom))
            img = img.resize((416, 416))
            results = self.model.predict(source=img)
            state = self.generate_state(results[0].boxes.data)
            if state != False:
                print(state)
        
    def train_model(self):
        sample = cv2.imread('boxhead-reinforcement-learning/image/game1.png')
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        print(sample.shape)
        pt1 = (332, 228)
        pt2 = (363, 273)
        cv2.rectangle(sample, pt1, pt2, color=(255,0,0), thickness=2)
        plt.imshow(sample)
        plt.show()

    def send_key(self):
        if self.temp:
            self.temp = False
            pag.press(['right', 'right'])
            self.temp = True

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
            self.get_image()
            
                


        time.sleep(10000)

boxhead = Boxhead()
def key_down(e):
    if e.name == 'command':
        boxhead.send_key()


keyboard.hook(key_down)
boxhead.main()
# boxhead.train_model()