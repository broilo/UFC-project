import urllib.request
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

#Capture the image we want
download = input('Please write khabib? ')

# Capture the number of images we want
n_images = int(input('How many images do you want? '))

# return the url link which contain the images
def url(download):
    url = 'https://www.google.com/search?tbm=isch&q='+download
    return url

# return the url link which contain the images
def extract_image(images):
    count = 0
    for image in images:
        #print(i['src'])
        try:
            #passing image urls one by one and downloading
            urllib.request.urlretrieve(image['src'], str(count)+".jpg")
            count += 1
            print("Number of images downloaded = "+str(count),end='\r')
        except Exception as e:
            pass

def image():
    #providing driver path
    chrome_path = r"C:\Users\Carlisson\Desktop\chromedriver\chromedriver.exe"
    driver = webdriver.Chrome(executable_path = chrome_path)

    site = url(download)

    #passing site url
    driver.get(site)

    #parsing
    soup = BeautifulSoup(driver.page_source, 'html.parser')

    #scraping image urls with the help of image tag and class used for images
    images = soup.find_all("img", attrs={'class':"rg_i Q4LuWd"},limit=n_images)

    extract_image(images)
    # closing web browser
    driver.close()

# run the code
image()


