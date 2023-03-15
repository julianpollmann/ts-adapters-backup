import bs4
import requests
from bs4 import BeautifulSoup
from math import ceil
from os import makedirs, path
from tqdm import tqdm

class DoiScraper:

    #data_dir = 'scraped_data'
    base_url = 'https://www.cochranelibrary.com/cdsr/reviews'
    dois = []
    filename = 'dois.txt'
    header = {
        "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
        "accept-encoding": "gzip, deflate",
        "accept-language": "en-US,en;q=0.9",
        "sec-fetch-dest": "document",
        "sec-fetch-mode": "navigate",
        "sec-fetch-site": "same-origin",
        "sec-fetch-user": "?1",
        "upgrade-insecure-requests": "1",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36"
    }
    URL = 'https://www.cochranelibrary.com/en/search?min_year=&max_year=&custom_min_year=&custom_max_year=&searchBy=6&searchText=*&selectedType=review&isWordVariations=&resultPerPage=25&searchType=basic&orderBy=relevancy&publishDateTo=&publishDateFrom=&publishYearTo=&publishYearFrom=&displayText=&forceTypeSelection=true&p_p_id=scolarissearchresultsportlet_WAR_scolarissearchresults&p_p_lifecycle=0&p_p_state=normal&p_p_mode=view&p_p_col_id=column-1&p_p_col_count=1&cur='

    def __init__(self, data_dir='scraped_data') -> None:
        self.data_dir = data_dir

    def get_doi(self, body: str) -> str:
        if body.find("a") is None:
            raise Exception("get_doi: no href found!")
        link = body.find("a")['href']
        doi = link[link.index('doi/')+4:link.index('/full')]

        return doi

    def scrape(self, results_per_page=25) -> list:
        client = requests.Session()
        client.headers.update(self.header)

        #determine total number of reviews (first get also gives us the necessary cookies for future queries)
        soup_search_page = BeautifulSoup(client.get(self.base_url).text, 'html.parser')
        num_reviews = int(soup_search_page.find("span", {"class": "results-number"}).contents[0].string)
        num_search_pages = ceil(num_reviews/results_per_page)

        #loop through the pages of results
        print('##### Reviews/Searchpages found: ' + str(num_reviews) + '/' + str(num_search_pages))
        for page in tqdm(range(num_search_pages)):

            if page % 50 == 0:
                #prevent timeout
                client = requests.Session()
                client.headers.update(self.header)
                client.get(self.base_url)
            
            soup = BeautifulSoup(client.get(self.URL + str(page+1)).text, 'html.parser')
            
            if soup.find("div", {"class": "search-results-section-body"}) is not None:
                for child in soup.find("div", {"class": "search-results-section-body"}).contents:
                    if type(child) == bs4.element.Tag and "search-results-item" in child['class']:
                        try:
                            body = child.find("div", {"class": "search-results-item-body"})
                            if body is None:
                                raise Exception('no body!')
                            self.dois.append(self.get_doi(body))
                        except:
                            pass

        self.write_to_file(dois=self.dois)

        return self.dois

    def write_to_file(self, dois: list) -> None:
        if not path.exists(f'{self.data_dir}'):
            makedirs(f'{self.data_dir}')      

        with open(path.join(self.data_dir, self.filename), 'a') as outfile:
            for doi in dois:
                outfile.write(doi + '\n')

    def read_from_file(self) -> list:
        with open(path.join(self.data_dir, self.filename), 'r') as infile:
            data = infile.read()
            dois = data.splitlines()

            return dois
