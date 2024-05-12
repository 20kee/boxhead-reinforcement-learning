
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver import ActionChains
import pyautogui as pag
import pyscreenshot as ImageGrab
import time
import base64

class Boxhead:
    def __init__(self):
        self.boxhead_url = 'https://flgamenara.tistory.com/6#google_vignette'
        self.driver = ''
        self.action = ''

        self.left_top = (24, 100)
        self.right_top = (738, 537)
    
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
        pag.click(350, 250)
        time.sleep(22)
        self.remove_ad()
        pag.click(393, 354)
        time.sleep(1)

        for _ in range(10):
            self.remove_ad()
            pag.click(646, 363)
            time.sleep(0.2)

        pag.click(443, 359)

    def game_restart(self):
        self.remove_ad()
        pag.click(381, 484)

    def get_image(self):
        while True:
            time.sleep(0.05)
            img = ImageGrab.grab(bbox=(*self.left_top, *self.right_top))
            img.save("boxhead.png")
        
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
        
        time.sleep(1)
        while True:
            self.remove_ad()
            
            self.game_start()
            
            self.get_image()
            
                


        time.sleep(10000)

boxhead = Boxhead()
boxhead.main()