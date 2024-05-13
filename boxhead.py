
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
import platform

class Boxhead:
    def __init__(self):
        self.boxhead_url = 'https://flgamenara.tistory.com/6#google_vignette'
        self.driver = ''
        self.action = ''

        if platform.system() == "Windows":
            self.left_top = (24, 100)
            self.right_bottom = (738, 537)
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

    def get_image(self):
        i = 0
        while True:
            i += 1
            time.sleep(2)
            img = ImageGrab.grab(bbox=(*self.left_top, *self.right_bottom))
            img.save("./boxhead-reinforcement-learning/images/{}.png".format(i))
        
    def train_model(self):
        sample = cv2.imread('boxhead-reinforcement-learning/image/game1.png')
        sample = cv2.cvtColor(sample, cv2.COLOR_BGR2RGB)
        print(sample.shape)
        pt1 = (332, 228)
        pt2 = (363, 273)
        cv2.rectangle(sample, pt1, pt2, color=(255,0,0), thickness=2)
        plt.imshow(sample)
        plt.show()

    def main(self):
        options = webdriver.ChromeOptions()
        options.add_experimental_option('excludeSwitches', ['enable-automation'])
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        self.action = ActionChains(self.driver)
        self.driver.get(self.boxhead_url)

        while True:
            try:
                ruffle = self.driver.find_element(By.TAG_NAME, 'ruffle-embed')
                self.driver.execute_script("window.scrollTo({}, {})".format(ruffle.location['x'], ruffle.location['y']))
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
boxhead.main()
# boxhead.train_model()