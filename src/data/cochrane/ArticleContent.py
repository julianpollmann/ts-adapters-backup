import json

from dataclasses import dataclass, asdict
from bs4 import BeautifulSoup

@dataclass
class ArticleContent:
    doi: str
    title: str
    is_free: bool
    language_links: dict
    content: dict

    def __init__(self, json_data: str) -> None:
        self.doi = json_data.get('doi')
        #self.url = json.get('url')
        self.title = json_data.get('title')
        self.is_free = json_data.get('is_free')
        self.language_links = json_data.get('language_links')
        if json_data.get('content'):
            self.content = json_data.get('content')
        else:
            self.content = {}

    def extract_title(self, soup: str) -> str:
        return soup.find("h1", {"class": "publication-title"}).string
    
    def extract_is_free(self, soup: str) -> str:
        return soup.find("div", {"class": "get-access-unlock"}) is None
    
    def get_language_links(self, soup: str) -> dict:
        languages = {}

        abstract_links = self.extract_language_links(soup=soup, doi=self.doi, part='abstract')
        pls_links = self.extract_language_links(soup=soup, doi=self.doi, part='pls')

        for key in abstract_links:
            if key in pls_links:
                languages.update({key: pls_links.get(key)})

        return languages

    def extract_language_links(self, soup: str, doi: str, part: str) -> dict:
        languages = {}
        if part == 'abstract':
            header = soup.find("section", {"class": "abstract"})
        else:
            header = soup.find("section", {"class": "pls"})

        languages.update({'en': self.url + doi})

        if header is not None:
            lang_nav = header.find("nav", {"class": "section-languages"})
            if lang_nav is not None:
                for a in lang_nav.findAll('a'):
                    if a.text != 'English':
                        iso_lang = self.map_languages(a.text)
                        languages.update({iso_lang: self.url + doi + '/' + a['href']})

        return languages

    def map_languages(self, lang: str):
        
        # Mapped to ISO 639-1
        languages = {
            'Bahasa Malaysia': 'ms',
            'Deutsch': 'de',
            'English': 'en',
            'Español': 'es',
            'Français': 'fr',
            'Hrvatski': 'hr',
            'Português': 'pt',
            'Русский': 'ru',
            'فارسی': 'fa',
            'ภาษาไทย': 'th',
            '日本語': 'ja',
            '简体中文': 'zh_hans',
            '繁體中文': 'zh_hant',
            '한국어': 'ko'
        }

        return languages.get(lang)

    def to_json(self):
        return {
            'doi': self.doi,
            'title': self.title,
            'is_free': self.is_free,
            'language_links': self.language_links,
            'content': self.content
        }