
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import pyautogui as pag
import time
import base64

class Boxhead:
    def __init__(self):
        self.boxhead_url = 'https://mukppe.tistory.com/1096'
        self.driver = ''

    def main(self):
        self.driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()))
        self.driver.get(self.boxhead_url)

        while True:
            try:
               self.driver.find_element(By.TAG_NAME, 'ruffle-embed')
               break
            except:
                pass
            
        ruffle = self.driver.find_element(By.TAG_NAME, 'ruffle-embed')
        print(ruffle.location)
        self.driver.execute_script("window.scrollTo({}, {})".format(ruffle.location['x'], ruffle.location['y']))
        ruffle.click()
        time.sleep(0.1)
        if len(self.driver.window_handles) == 2:
            self.driver.switch_to.window(self.driver.window_handles[-1])
            self.driver.close()
            self.driver.switch_to.window(self.driver.window_handles[0])
            ruffle = self.driver.find_element(By.TAG_NAME, 'ruffle-embed')
            ruffle.click()


        ruffle.click()
        time.sleep(10000)

if __name__ == "__main__":
    boxhead = Boxhead()
    boxhead.main()