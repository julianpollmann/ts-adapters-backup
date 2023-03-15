import json
import requests
from bs4 import BeautifulSoup
from os import makedirs, path, listdir
from tqdm import tqdm
from Content import Content
from Article import Article
from ArticleContent import ArticleContent

class ArticleScraper():
    base_url = 'https://www.cochranelibrary.com/cdsr/reviews'
    URL = 'https://www.cochranelibrary.com/cdsr/doi/'
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
    article_dir = 'articles'
    json_dir = 'json'
    withdrawn_fname = 'withdrawn.txt'

    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.create_data_dirs()

    def scrape_articles(self, dois: list):
        client = requests.Session()
        client.headers.update(self.header)
        client.get(self.base_url)

        for i, doi in enumerate(tqdm(dois)):
            if i > 0 and i % 50 == 0:
                client = requests.Session()
                client.headers.update(self.header)
                client.get(self.base_url)

            try:
                soup = BeautifulSoup(client.get(self.URL + doi).text, 'html.parser')

                self.save_html(doi=doi, soup=soup)

                article = Article(doi=doi, url=self.URL)
                article.title = article.extract_title(soup=soup)
                article.is_free = article.extract_is_free(soup=soup)
                article.language_links = article.get_language_links(soup=soup)

                self.save_json(article=article)

            except Exception as e:
                print(f'##### ERROR DOI {doi}: {e}')

                with open(path.join(self.data_dir, self.withdrawn_fname), 'a+') as f:
                    f.write(doi + '\n')

    def scrape_content(self, articles: list):
        for i, json_article in enumerate(tqdm(articles)):
           if not json_article.get('content'):
                client = requests.Session()
                client.headers.update(self.header)

                article = ArticleContent(json_data=json_article)

                # TODO get english lang, extract pls len and save on en language
                # Compare in language loop with other languages, if equal -> use index based approach, otherwise based on tags
                # in Content, different methods for languages

                
                for language, link in tqdm(article.language_links.items()):
                    try:
                        soup = BeautifulSoup(client.get(link).text, 'html.parser')

                        content = Content()
                        lang_content = content.add_abstract(language=language, soup=soup)

                        article.content.update(lang_content)
                    except Exception as e:
                        print(f'##### ERROR DOI: {json_article["doi"]}: {e}')

                        with open(path.join(self.data_dir, self.withdrawn_fname), 'a+') as f:
                            f.write(json_article.get('doi') + '\n')

                self.save_json(article)
           else:
               print(f"### Article already scraped: {json_article.get('doi')}")

    def read_json(self) -> list:
        articles = []
    
        for articles_fname in listdir(path.join(self.data_dir, 'json')):
            with open(path.join(self.data_dir, 'json', articles_fname), encoding='utf-8') as fh:
                articles.append(json.load(fh))

        return articles
    
    def read_json_by_dois(self, dois: list):
        articles = []

        for doi in dois:
            file_doi = '-'.join(doi.split('/'))

            with open(path.join(self.data_dir, 'json', file_doi + '.json'), encoding='utf-8') as fh:
                articles.append(json.load(fh))

        return articles
    
    def save_html(self, doi: str, soup: str) -> None:
        #write the retrieved html to a file for record-keeping purposes
        with open(path.join(self.data_dir, self.article_dir, '%s.html' % doi.replace('/', '-')), 'w') as f:
            f.write(str(soup))

    def save_json(self, article: Article) -> None:
        with open(path.join(self.data_dir, self.json_dir, '%s.json' % article.doi.replace('/', '-')), 'w', encoding='utf-8') as f:
            json.dump(article.to_json(), f, ensure_ascii=False, indent=2)
            #f.write(json.dumps(article.to_json(), indent=2))

    def save_merged_json(self) -> None:
        # now create single json file with all the articles
        articles = self.read_json()

        with open(path.join(self.data_dir, 'data.json'), 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)

    def create_data_dirs(self) -> None:
        if not path.exists(self.data_dir):
            makedirs(f'{self.data_dir}')

        if not path.exists(path.join(self.data_dir, self.article_dir)):
            makedirs(path.join(self.data_dir, self.article_dir))

        if not path.exists(path.join(self.data_dir, self.json_dir)):
            makedirs(path.join(self.data_dir, self.json_dir))