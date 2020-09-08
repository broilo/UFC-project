#scrit to scrape fighters data from UFC website

import requests
from bs4 import BeautifulSoup
import pandas as pd



def ranked(soup):
    
    if len(soup.string.split()) == 6:
        return True
    else:
        return False
        
def get_ranking_and_category(soup):
    
    infos = soup.string.split()
    if ranked(soup):        
        return infos[:2]
    else:
       if infos[1] == 'Camp.':
           return ['Champion'] + [infos[0]]
       else:
           return ['-'] + [infos[0]]
        

def get_card(soup):
    infos = soup.string.split()
    
    if ranked(soup):        
        return infos[4].split('-')
    else:
       return infos[3].split('-')
   
def get_height_weight_age(soup):
    '''sends soup_bio1'''
    row = []
    for info in soup:
        if info.string != None:
            row.append(info.string)
            
    soup_age = soup[0].find('div', 
                          class_="field field--name-age field--type-integer field--label-hidden field__item")
    row.append(soup_age.string)       
    return row


def get_date_wingspan_legspan(soup):
    '''sends soup_bio2'''
    row = []    
    for info in soup:
        row.append(info.string)
    return row

def get_soup(fighter_name):
    fighter_names = fighter_name.lower().split()
    url = "https://www.ufc.com.br/athlete/{}-{}".format(fighter_names[0],
                                                        fighter_names[1])    
    results = requests.get(url)
    src = results.content
    soup = BeautifulSoup(src, 'lxml')
    
    return soup
    
def get_soups(soup):    
    #go to the html section which contains all the card information
    soup_card = soup.find('div', 
                          class_="c-hero__headline-suffix tz-change-inner")
    
    #go to the html section which contains all the information related 
    #to fighter body and start day:
    soup_body = soup.find_all('div', class_="c-bio__row--3col")
    #the infos are in two different rows
    soup_body_1 = soup_body[0].find_all('div', class_='c-bio__text')
    soup_body_2 = soup_body[1].find_all('div', class_='c-bio__text') 
    
    return [soup_card, soup_body_1, soup_body_2]


def get_row(fighter_name, df = None):
    
    if df is None:
        df = pd.DataFrame(columns=['Nome', 'Ranking', 'Category', 'Vitoria', 
                           'Derrota', 'Empate', 'Altura', 'Peso', 'Idade', 
                           'Data de início', 'Envergadura', 
                           'Envergadura da perna'])
    soups = get_soups(get_soup(fighter_name))
    
    row = [fighter_name.title()]
    row += get_ranking_and_category(soups[0])
    row += get_card(soups[0])
    row += get_height_weight_age(soups[1])
    row += get_date_wingspan_legspan(soups[2])

    df = df.append(pd.Series(row, index=df.columns), ignore_index=True)
    
    return df


