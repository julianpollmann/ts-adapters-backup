from dataclasses import dataclass, field
from bs4 import BeautifulSoup
import re

@dataclass
class Content:
    doi: str
    title: str
    is_free: bool
    language_links: dict
    content: dict

    def __init__(self) -> None:
        pass
        #self.language = language
        #self.soup = soup
        # self.doi = json.get('doi')
        # self.title = json.get('title')
        # self.is_free = json.get('is_free')
        # self.language_links = json.get('language_links')
        # self.content = json.get('content')

    def add_abstract(self, language: str, soup: str):
        pls = self.extract_pls_div(soup=soup)
        pls_type = self.extract_pls_type(pls=pls)

        return {
            language: {
                'title': self.extract_title(soup=soup),
                'abstract': self.extract_abstract(soup=soup),
                'pls_title': self.get_text(pls.find("h3")),
                'pls_type': pls_type,
                'pls': self.extract_pls(pls_type=pls_type, pls=pls)
            }
        }

    def extract_title(self, soup: str) -> str:
        return soup.find("h1", {"class": "publication-title"}).string
    
    def extract_abstract(self, soup: str) -> list:
        sections = []

        #go heading by heading through the abstract
        abstract = soup.find("div", {"class": "full_abstract"})
        if abstract is None:
            raise Exception("abstract not found!")

        for section in abstract("section"):
            sec_object = {}
            sec_object['heading'] = self.get_text(section.find("h3", {"class": "title"}))
            text = [self.get_text(para) for para in section("p")]
            sec_object['text'] = '\n'.join(text)
            sections.append(sec_object)

        return sections
    
    def extract_pls_div(self, soup: str) -> str:
        #do the same for the plain-language summary
        pls = soup.find("div", {"class": "abstract_plainLanguageSummary"})
        if pls is None:
            raise Exception("pls not found!")
        
        return pls
    
    def extract_pls_type(self, pls: str):
        # determine the type of pls: "sectioned" or "long"
        if re.search(r"<p>(<i>)?<b>", str(pls)):
            return 'sectioned'
        else:
            return 'long'
        #pls_type = 'long'

        #if pls.find("b") is not None:
        #    pls_type = 'sectioned'
        #else:
        #    pls_type = 'long'

        #return pls_type
    
    def extract_pls(self, pls_type: str, pls: str):
        sections = []

        if pls_type == 'sectioned':

            heading_indices = []
            texts = []

            for para in pls("p"):
                # Only process non empty p tags
                if len(para.get_text(strip=True)) != 0:
                    # Process subheadings, wrapped in bold tags
                    if re.search(r"<p>(<i>)?<b>", str(para)):
                    #if para.find("b") is not None:
                        heading = self.get_text(para.find("b"))
                        if heading[-1] == ':':
                            heading = heading[:-1]
                        texts.append(heading)
                        heading_indices.append(len(texts)-1)

                        #now grab text if there is some in the same paragraph as the heading
                        text_list = list(para.strings)
                        if len(text_list) > 1 and len(''.join(text_list[1:]).strip()) > 0:
                            text = self.get_text_gen(text_list[1:])
                            texts.append(text)
                    else:
                        texts.append(self.get_text(para))

            #edge case, if there is text before the first heading
            if heading_indices[0] > 0:
                sections.append({'heading': '', 'text': '\n'.join(texts[:heading_indices[0]])})
            
            for i in range(len(heading_indices)-1):
                sections.append({'heading': texts[heading_indices[i]], 'text': '\n'.join(texts[heading_indices[i]+1:heading_indices[i+1]])})

            #we know that there is at least 1 heading, so no empty list check
            sections.append({'heading': texts[heading_indices[-1]], 'text': '\n'.join(texts[heading_indices[-1]+1:])})
        else:
            text = [self.get_text(para) for para in pls("p")]
            sections.append('\n'.join(text))

        return sections

    def get_text(self, para):
        #first replace <br> with newlines
        soup = BeautifulSoup(str(para).replace('<br/>', '\n').replace('\n ', '\n'), 'html.parser')
        text = ''.join(soup.strings).strip()

        #replace Unicode hyphen with regular '-'
        text = text.replace('\u2010', '-')

        return text
    
    def get_text_gen(self, gen) -> str:
        gen = [g.strip() for g in gen]
        text = ''.join(gen).strip()
        if len(text) > 0 and text[0] == ':':
            text = text[1:].strip()
        text = text.replace('\u2010', '-').strip()
        text = text.replace('\u2013', '-').strip()

        return text

    def to_json(self):
        return {
            'doi': self.doi,
            'title': self.title,
            'is_free': self.is_free,
            'language_links': self.language_links,
            'content': self.content
        }