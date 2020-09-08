from urllib.request import urlopen
import requests
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver

PATH_TO_SAVE = '/home/broilo/Documents/GitHub/Dataset/UFC-project/data-mining/'


def get_rows(name):
    """[summary]

    Args:
        name ([type]): [description]

    Returns:
        [type]: [description]
    """

    # Catch the HTML content from URL
    wiki_url = "https://pt.wikipedia.org/wiki/"+name
    response = requests.get(wiki_url)

    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', attrs={'class': 'wikitable sortable'})

    rows = table.find_all('tr')
    return rows

# print(get_rows(name))


def dataframe(name):
    """[summary]

    Args:
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Builds a dataframe form the HTML content
    rows = get_rows(name)
    columns = [v.text.replace('\n', '') for v in rows[0].find_all('th')]

    df = pd.DataFrame(columns=columns)

    for i in range(2, len(rows)):
        tds = rows[i].find_all('td')

        values = [td.text.replace('\n', '') for td in tds]
        df = df.append(pd.Series(values, index=columns), ignore_index=True)
    return df


def save_dataframe(name):
    """[summary]

    Args:
        name ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Save the dataframe
    df = dataframe(name)
    df.to_csv(PATH_TO_SAVE + name + '.csv', index=False)
    return df


def main():
    fighters = ['Conor McGregor', 'Khabib Nurmagomedov',
                'Israel Adesanya', 'Paulo Costa',
                'Dominick Reyes', 'Jan Blachowicz',
                'Claudia Gadelha', 'Yan Xiaonan',
                'Hakeem Dawodu', 'Zubaira Tukhugov',
                'Kai Kara-France', 'Brandon Royval']  # change your set of queries here
    queries = fighters
    for query in queries:
        save_dataframe(query)


if __name__ == "__main__":
    main()
