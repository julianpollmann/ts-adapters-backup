import argparse
import copy
import json
from os import path, makedirs
from nltk.tokenize import sent_tokenize

def abs_length(article):
    return sum([len(x["text"]) for x in article["abstract"]])


def truncate_articles_old(data: list):
    # truncate abstract to only main results onwards
    articles = []
    for article in data:
        for language, content in article["content"].items():
            first_index = -1
            for index, section in enumerate(content["abstract"]):
                if "main result" in section["heading"].strip().lower():
                    first_index = index
                    break
            article["content"][language]["abstract"] = article["content"][language][
                "abstract"
            ][first_index:]
        articles.append(article)

    return articles


def remove_short_abstracts(data: list) -> list:
    """Removes short abstracts with characater counts < 1000

    Keyword arguments:
    data -- list of articles
    """

    articles = []

    for article in data:
        if article.get("content"):
            modified_content = {}

            for lang, content in article.get("content").items():
                if abs_length(content) >= 1000:
                    modified_content[lang] = content

            article["content"] = modified_content

            articles.append(article)

    return articles


def get_articles(data: list, searched_type: str):
    articles = []
    for article in data:
        content = article.get("content")
        pls_type = next(iter(content.values())).get("pls_type")
        if pls_type == searched_type:
            articles.append(article)

    return articles


def one_para_filter(text: str, language: str):
    keywords = {
        "en": ["review", "journal", "study", "studies", "paper", "trial"],
        "es": ["revisión", "estudio", "estudios"],
        "de": ["review", "studie", "studien"],  # evidenz?
        "fa": ["مطالعه", "مطالعات", "آزمایش", "مرور", "مرور"],
        "fr": [
            "revue",
            "objectif",
            "résultats",
            "rapport",
            "étude",
        ],  # effets = Effekte
        "pt": [
            "revisão",
            "revista",
            "periódico",
            "estudo",
            "estudos",
            "jornal",
            "teste",
            "revisa",
        ],
        "zh_hans": ["评论", "审查", "篇评论", "综述", "审讯", "考验", "试验"],
        "zh_hant": ["個研究", "的研究", "回顧"],
        "ko": ["문헌고찰", "연구", "검토", "신문", "공부하다", "연구"],
        "th": ["ทบทวน", "การทดลอง", "วารสาร", "ศึกษา", "การศึกษา", "กระดาษ"],
        "ja": ["レビュー", "ジャーナル", "勉強", "研究", "試み", "試し", "試験"],
        "ru": [],
    }

    sentence_list = sent_tokenize(text)
    first_index = -1
    if keywords.get(language):
        for index, sentence in enumerate(sentence_list):
            if any(word in sentence.lower() for word in keywords.get(language)):
                first_index = index
                break
        return " ".join(sentence_list[first_index:]) if first_index > -1 else ""
    else:
        return ""


def res_para(text: str, language: str):
    keywords = {
        "en": ["journal", "study", "studies", "trial"],
        "es": [
            "revisión",
            "estudio",
            "estudios",
            "investigación",
            "resultado",
            "resultados",
            "ensayo"
        ],
        "de": ["review", "reviews", "studie", "studien"],  # evidenz?
        "fa": ["مطالعات", "مطالعه", "مرور", "مجله"],
        "fr": [
            "revue",
            "objectif",
            "résultats",
            "rapport",
            "essais",
            "étude",
        ],  # effets = Effekte; éval(ué)/Evaluation? essais = Paper; analyses?
        "pt": ["revista", "estudo", "estudos", "teste"],
        "zh_hans": ["综述", "学习", "杂志", "审判"],
        "zh_hant": ["回顧", "篇研究", "系統性審閱", "審查", "的研究"],
        "ko": ["본 고찰은", "신문", "공부하다", "연구", "재판"],
        "th": ["วารสาร", "ศึกษา", "การศึกษา", "การทดลอง"],
        "ja": ["ジャーナル", "勉強", "研究"],
        "ru": [],
    }

    first_index = -1
    sentence_list = sent_tokenize(text)
    if keywords.get(language):
        for index, sentence in enumerate(sentence_list):
            if any(word in sentence.lower() for word in keywords.get(language)):
                first_index = index
                break
        return first_index > -1 and (index + 1) / len(sentence_list) <= 0.5
    else:
        return ""


def res_heading(heading: str, language: str):
    keywords = {
        "en": [
            "find",
            "found",
            "evidence",
            "tell us",
            "study characteristic",
            "was studied",
        ],
        "es": [
            "características de los estudios",
            "criterios de selección",
        ],  # Resultados clave; evidencia; Criterios de selección; principales resultados; ¿Qué se estudió en la revisión?
        "de": [
            "herausgefunden",
            "gefunden",
            "fanden",
            "finden",
            "untersucht",
            "Studien",
            "merkmal",
            "studiencharakteristiken",
        ],  # Auswahlkriterien? #Datenerhebung und -analyse?; Ergebnisse?, Was wir getan haben? Evidenz? Was wir in diesem Review untersuchten?
        "fr": [
            "étudié",
            "étudient",
            "caractéristiques des études",
            "caractéristique des études",
            "Caractéristiques de l’étude",
            "données probantes",
        ],  # Méthodes; Critères de sélection; résultats; Recueil et analyse des données - Analyse des gegebenen; Comment cette revue a-t-elle été réalisée - Wie hat diese Studie das realisiert?
        "fa": [
            "پیدا کردن",
            "یافت",
            "شواهد و مدارک",
            "به ما بگو",
            "ویژگی مطالعه",
            "ویژگی‌های مطالعه",
            "شواهد و مدارک",
            "شواهدی را",
        ],
        "pt": [
            "características dos estudos",
            "características do estudo",
            "foi estudado",
            "encontrar",
            "encontrad",
            "evidência",
            "nos digam",
        ],
        "zh_hans": ["寻找", "成立", "证据", "告诉我们", "学习特点", "研究特征"],
        "zh_hant": ["尋找", "成立", "證據", "告訴我們", "學習特點", "研究特徵", "研究特性", "研究特點"],
        "ko": ["찾다", "설립하다", "증거", "우리에게 말해줘", "연구 특성", "연구 특징"],
        "th": [
            "หา",
            "พบ",
            "หลักฐาน",
            "บอกพวกเรา",
            "ลักษณะการศึกษา",
            "ลักษณะของการศึกษา",
        ],
        "ja": [
            "探す",
            "見つかった",
            "証拠",
            "教えて",
            "研究の特徴",
            "試験の特性",
            "なエビデンスが",
            "試験の特徴",
            "研究の特性",
        ],
        "ru": [],
    }
    if language not in keywords.keys():
        print("language: " + language + " not in keywords.keys()")

    return any(word in heading.lower() for word in keywords.get(language))


def pls_length(article):
    # if article["pls_type"] == "long":
    #     return len(article["pls"][0]["text"])
    # else:
    return sum([len(x["text"]) for x in article["pls"]])


# def truncate_abstracts(data: list, language: str) -> list:
#     # truncate abstract to only main results onwards
#     keywords = {
#         "en": "main result",
#         "es": "resultados principales",  # resultados principales
#         "de": "ergebnis",  # Hauptergebnisse / Wesentliche Ergebnisse
#         "fr": "résultats principaux",  # résultats principaux / Principaux résultats
#         "fa": "نتایج اصلی",
#         "pt": "principais resultados",
#         "zh_hans": "主要结果",
#         "zh_hant": "主要結果",
#         "ko": "주요 결과",
#         "th": "ผลลัพธ์หลัก",
#         "ja": "主な結果",
#     }
#     for article in data:
#         first_index = -1
#         for index, section in enumerate(article["abstract"]):
#             keyword = keywords.get(language)

#             if keyword in section["heading"].strip().lower():
#                 first_index = index
#                 break
#         article["abstract"] = article["abstract"][first_index:]

#     return data

def truncate_abstracts(data: list, truncate_en_only: bool) -> list:
    """Truncate the abstracts to only main results onwards

    For all abstracts with equal len, index-based method is used; keyword-based matching for other cases
    Gets the index for the first english abstract paragraph containing 'main result' in heading
    If no english abstract available, this article will be left out

    Keyword arguments:
    data -- list of articles
    truncate_en_only -- if True only main language (english) will be truncated, of False all languages.
    """

    articles = []
    for article in data:
        first_index = -1

        # and article.get('content').get('en') and article.get('content').get('en').get('abstract')
        if article.get("content") and article.get("content").get("en"):
            en_content = article.get("content").get("en")
            en_abs_len = len(en_content.get("abstract"))

            for index, section in enumerate(en_content.get("abstract")):
                if "main result" in section.get("heading").strip().lower():
                    first_index = index
                    break

            if truncate_en_only:
                # Truncate english abstracts only
                article["content"]["en"]["abstract"] = article["content"]["en"]["abstract"][first_index:]

                for lang, content in article.get("content").items():
                    if lang != "en":
                        article["content"][lang]["abstract"] = article["content"][lang]["abstract"]
            else:
                # Truncate all abstracts
                for language, content in article.get("content").items():
                    if len(content.get("abstract")) == en_abs_len:
                        # resolve by index
                        article["content"][language]["abstract"] = article["content"][language]["abstract"][first_index:]
                    else:
                        # resolve by keyword
                        first_lang_index = -1
                        keywords = {
                            "ms": "keputusan",
                            "de": "ergebnisse",
                            "es": "resultados principales",
                            "fr": "résultats principaux",
                            "hr": "rezultati",
                            "pt": "principais resultados",
                            "ru": "основные результаты",
                            "fa": "نتایج اصلی",
                            "th": "ผลการวิจัย",
                            "ja": "主な結果",
                            "zh_hans": "主要结果",
                            "zh_hant": "主要結果",
                            "ko": "주요 결과",
                        }

                        for index, section in enumerate(content.get("abstract")):
                            if (keywords.get(language) in section.get("heading").strip().lower()):
                                first_lang_index = index
                                break

                        article["content"][language]["abstract"] = article["content"][language]["abstract"][first_lang_index:]

            # for language in article.get('content'):
            #    article['content'][language]['abstract'] = article['content'][language]['abstract'][first_index:]

            articles.append(article)

    return articles

# def split_by_language(data_dir: str):
#     data = json.load(open(path.join(data_dir, "data.json")))

#     languages = {}
#     for item in data:
#         for lang, content in item.get("content").items():
#             create_language_dirs(data_dir=data_dir, language=lang)

#             language_item = {
#                 "doi": item.get("doi"),
#                 "title": content.get("title"),
#                 "is_free": item.get("is_free"),
#                 "language": lang,
#                 "abstract": content.get("abstract"),
#                 "pls_title": content.get("pls_title"),
#                 "pls_type": content.get("pls_type"),
#                 "pls": content.get("pls")
#                 if content.get("pls_type") == "sectioned"
#                 else content.get("pls")[0],
#             }

#             if lang in languages:
#                 languages[lang].append(language_item)
#             else:
#                 languages[lang] = [language_item]

#     for lang, articles in languages.items():
#         with open(
#             path.join(data_dir, "languages", lang, "data.json"), "w", encoding="utf-8"
#         ) as f:
#             json.dump(articles, f, ensure_ascii=False, indent=2)


def create_language_dirs(data_dir: str, language: str):
    if not path.exists(path.join(data_dir, "languages", language)):
        makedirs(path.join(data_dir, "languages", language))

def get_pls_len(article: dict) -> dict:
    if article.get("content"):
        for lang, content in article.get("content").items():
            article["content"][lang]["pls_len"] = len(content.get("pls"))

    return article

def get_abstract_len(article: dict) -> dict:
    if article.get("content"):
        for lang, content in article.get("content").items():
            article["content"][lang]["pls_len"] = len(content.get("pls"))

    return article

def remove_empty_pls(data: list) -> list:
    """Remove languages where no pls text is present

    Keyword arguments:
    data -- list of articles
    """

    articles = []

    for article in data:
        content_data = {}

        for lang, content in article.get("content").items():
            if content.get("pls") and content.get("pls")[0] != "":
                content_data[lang] = content

        article["content"] = content_data
        articles.append(article)

    return articles

def save_articles(output_dir: str, data: list, language: str = None) -> None:
    if not path.exists(path.join(output_dir)):
        makedirs(path.join(output_dir))

    if language:
        filename = path.join(f"{output_dir}/processed_data_{language}.json")
    else:
        filename = path.join(f"{output_dir}/processed_data.json")

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_languages(articles: list) -> dict:
    # Get all used languages
    languages = {}
    for article in articles:
        if article.get("content"):
            for lang, content in article.get("content").items():
                languages[lang] = []

    return languages

def sort_articles_into_languages(articles: list, languages: dict) -> dict:
    # Sort articles into languages
    for language in languages:
        for article in articles:
            if language in article.get("content"):
                modified_article = copy.deepcopy(article)
                modified_article["content"] = {language: article.get("content").get(language)}
                languages[language].append(modified_article)


    return languages

def remove_articles_without_content(data: list) -> list:
    return [article for article in data if article.get("content")]

def add_paragraph_pos(data: list) -> list[dict]:
    """Add paragraph position to each paragraph"""
    for article in data:
        for content in article.get("content").values():
            for index, para in enumerate(content.get("abstract")):
                para["para_pos"] = index

            for index, para in enumerate(content.get("pls")):
                para["para_pos"] = index

    return data

def truncate_pls_content(content: dict, lang: str) -> dict:
    # LONG PLS
    if content["pls_type"] == "long":
        pls_text = content["pls"][0]["text"]
        para_count = len(pls_text.strip().split("\n"))

        # LONG Single Paragraph
        if para_count == 1:
            content["pls_subtype"] = "single"
            content["pls"][0]["text"] = one_para_filter(text=pls_text, language=lang)
        
        # LONG Multi Paragraphs
        if para_count > 1:
            content["pls_subtype"] = "multi"

            first_index = -1
            for index, para in enumerate(pls_text.strip().split("\n")):
                if res_para(text=para, language=lang):
                    first_index = index
                    break

            if first_index > -1:
                content["pls"][0]["text"] = "\n".join(pls_text.strip().split("\n")[first_index:])
            else:
                content["pls"] = []

    # SECTIONED PLS
    if content["pls_type"] == "sectioned":
        first_index = -1
        for index, section in enumerate(content['pls']):
            if res_heading(section['heading'], language=lang):
                first_index = index
                break
        
        if first_index > -1:
            content['pls'] = content['pls'][first_index:]
        else:
            content['pls'] = []

    return content

def truncate_pls(data: list, truncate_en_only: bool) -> list:
    """Truncates pls based on keywords"""

    articles = []
    for article in data:
        for lang, content in article.get("content").items():
            if truncate_en_only:
                if lang == "en":
                    article["content"][lang] = truncate_pls_content(content=content, lang=lang)
            else:
                article["content"][lang] = truncate_pls_content(content=content, lang=lang)

        articles.append(article)

    return articles

def filter_by_language(articles: list[dict], language: str) -> list[dict]:
    """Filters articles by language."""
    modified_articles = []

    for article in articles:
        if language in article.get("content"):
            modified_content = {language: article.get("content").get(language)}
            article["content"].clear()
            article["content"] = modified_content
            modified_articles.append(article)

    return modified_articles

def trim_content(modified_content: dict, lang: str, content: dict) -> dict:
    if content["pls_type"] == "long":
        if content["pls_subtype"] == "single":
            if pls_length(content)/abs_length(content) >= 0.20 and pls_length(content)/abs_length(content) <= 1.4:
                modified_content[lang] = content
        if content["pls_subtype"] == "multi":
            if pls_length(content)/abs_length(content) >= 0.30 and pls_length(content)/abs_length(content) <= 1.3:
                modified_content[lang] = content
    if content["pls_type"] == "sectioned":
        if pls_length(content)/abs_length(content) >= 0.30 and pls_length(content)/abs_length(content) <= 1.3:
            modified_content[lang] = content

    return modified_content

def trim_by_ratio(data: list, truncate_en_only: bool) -> list:
    """Trims each language content based on ratio between pls <-> abstract len."""

    articles = []

    for article in data:
        modified_content = {}

        if truncate_en_only:
            lang = "en"
            content = article.get("content").get(lang)
            if content:
                modified_content = trim_content(modified_content, lang, content)

            for lang, content in article.get("content").items():
                if lang != "en":
                    modified_content[lang] = content
        else:
            for lang, content in article.get("content").items():
                modified_content = trim_content(modified_content, lang, content)
            
        article["content"] = modified_content
        articles.append(article)

    return articles

def reformat_pls(data: list) -> list:
    """Reformat long pls to match sectioned pls format"""

    for article in data:
        for lang, content in article.get("content").items():
            if content["pls_type"] == "long":
                article["content"][lang]["pls"] = [{"text": content["pls"][0]}]

    return data

def drop_small_sample_size(data: list) -> list:
    """Drop languages with less than 100 articles"""
    languages = get_languages(articles=data)

    # Get Article Count for Each Language
    for article in data:
        for lang in languages:
            if lang in article.get("content"):
                languages[lang].append(article)
    
    # Get Languages to remove
    languages_to_remove = [lang for lang in languages if len(languages[lang]) < 100]

    # Remove Languages
    for article in data:
        modified_content = {}
        for lang, content in article.get("content").items():
            if lang not in languages_to_remove:
                modified_content[lang] = content
        article["content"] = modified_content

    return data

def clean_up_data(data_dir: str, lang: str, output_dir: str, sample: int, truncate_en_only: bool):
    articles = json.load(open(path.join(data_dir, "data.json")))

    articles = remove_articles_without_content(data=articles)
    articles = remove_empty_pls(data=articles)

    if lang:
        articles = filter_by_language(articles=articles, language=lang)

    articles = reformat_pls(data=articles)
    articles = add_paragraph_pos(data=articles)

    articles = truncate_abstracts(data=articles, truncate_en_only=truncate_en_only)
    articles = remove_short_abstracts(data=articles)

    articles = truncate_pls(data=articles, truncate_en_only=truncate_en_only)
    articles = remove_empty_pls(data=articles)

    articles = trim_by_ratio(data=articles, truncate_en_only=truncate_en_only)
    articles = remove_articles_without_content(data=articles)
    articles = drop_small_sample_size(data=articles)

    if truncate_en_only:
         # Remove articles without english content, since no reference is available
        articles = [article for article in articles if "en" in article.get("content")]

    if sample:
        articles = articles[:sample]

    languages = get_languages(articles=articles)
    languages = sort_articles_into_languages(articles=articles, languages=languages)

    save_articles(output_dir=output_dir, data=articles)

    for lang, sorted_articles in languages.items():
        print(f"Saving {lang} articles ({len(sorted_articles)} articles)")
        save_articles(output_dir=output_dir, data=sorted_articles, language=lang)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="Cochrane Preprocessing", description="Preprocessing Cochrane Data")
    parser.add_argument("--sample", type=int, help="Sample No of articles")
    parser.add_argument("--language", type=str, help="Language to process")
    parser.add_argument("--truncate_en_only", type=bool, default=True, help="Truncates only english articles. Setting to False will truncate all languages by keywords.")
    parser.add_argument("--data_dir", type=str, default="./scraped_data", help="Directory of scraped data")
    parser.add_argument("--output_dir", type=str, default="./processed_data", help="Output directory")

    args = parser.parse_args()

    clean_up_data(data_dir=args.data_dir, lang=args.language, output_dir=args.output_dir, sample=args.sample, truncate_en_only=args.truncate_en_only)
