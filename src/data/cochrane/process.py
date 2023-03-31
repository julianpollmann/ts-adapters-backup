import argparse
from nltk.tokenize import sent_tokenize
import json
from os import path, makedirs

parser = argparse.ArgumentParser(
    prog="Cochrane Scraper", description="Scraper for Cochrane reviews"
)
parser.add_argument(
    "--type",
    type=str,
    nargs=1,
    default="dois",
    help="Which data type should be scraped (dois or reviews?)",
)
parser.add_argument(
    "--dois", type=str, nargs=1, default="dois", help="Path of filelist with dois"
)
parser.add_argument(
    "--results_per_page",
    type=str,
    nargs=1,
    default="dois",
    help="Path of filelist with dois",
)
parser.add_argument(
    "--data_dir", type=str, default="./scraped_data", help="Directory of scraped data"
)
args = parser.parse_args()


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
    if article["pls_type"] == "long":
        return len(article["pls"])
    else:
        return sum([len(x["text"]) for x in article["pls"]])


def truncate_abstracts(data: list, language: str) -> list:
    # truncate abstract to only main results onwards
    keywords = {
        "en": "main result",
        "es": "resultados principales",  # resultados principales
        "de": "ergebnis",  # Hauptergebnisse / Wesentliche Ergebnisse
        "fr": "résultats principaux",  # résultats principaux / Principaux résultats
        "fa": "نتایج اصلی",
        "pt": "principais resultados",
        "zh_hans": "主要结果",
        "zh_hant": "主要結果",
        "ko": "주요 결과",
        "th": "ผลลัพธ์หลัก",
        "ja": "主な結果",
    }
    for article in data:
        first_index = -1
        for index, section in enumerate(article["abstract"]):
            keyword = keywords.get(language)

            if keyword in section["heading"].strip().lower():
                first_index = index
                break
        article["abstract"] = article["abstract"][first_index:]

    return data


def split_by_language(data_dir: str):
    data = json.load(open(path.join(data_dir, "data.json")))

    languages = {}
    for item in data:
        for lang, content in item.get("content").items():
            create_language_dirs(data_dir=data_dir, language=lang)

            language_item = {
                "doi": item.get("doi"),
                "title": content.get("title"),
                "is_free": item.get("is_free"),
                "language": lang,
                "abstract": content.get("abstract"),
                "pls_title": content.get("pls_title"),
                "pls_type": content.get("pls_type"),
                "pls": content.get("pls")
                if content.get("pls_type") == "sectioned"
                else content.get("pls")[0],
            }

            if lang in languages:
                languages[lang].append(language_item)
            else:
                languages[lang] = [language_item]

    for lang, articles in languages.items():
        with open(
            path.join(data_dir, "languages", lang, "data.json"), "w", encoding="utf-8"
        ) as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)


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


def remove_articles_without_content(data: list) -> list:
    articles = []
    for article in data:
        if article.get("content"):
            articles.append(article)

    return articles


def truncate_articles(data: list) -> list:
    """Truncate the abstracts to only main results onwards

    For all abstracts with equal len, index-based method is used; keyword-based matching for other cases
    Gets the index for the first english abstract paragraph containing 'main result' in heading
    If no english abstract available, this article will be left out

    Keyword arguments:
    data -- list of articles
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

            for language, content in article.get("content").items():
                if len(content.get("abstract")) == en_abs_len:
                    # resolve by index
                    article["content"][language]["abstract"] = article["content"][
                        language
                    ]["abstract"][first_index:]
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
                        if (
                            keywords.get(language)
                            in section.get("heading").strip().lower()
                        ):
                            first_lang_index = index
                            break

                    article["content"][language]["abstract"] = article["content"][
                        language
                    ]["abstract"][first_lang_index:]

            # for language in article.get('content'):
            #    article['content'][language]['abstract'] = article['content'][language]['abstract'][first_index:]

            articles.append(article)

    return articles


def remove_empty_pls(data: list) -> list:
    """Remove languages where no pls text is present

    Keyword arguments:
    data -- list of articles
    """

    articles = []

    for article in data:
        content_data = {}

        for lang, content in article.get("content").items():
            if content.get("pls")[0] != "":
                content_data[lang] = content

        article["content"] = content_data
        articles.append(article)

    return articles


def save_articles(language: str, data: list) -> None:
    if not path.exists(path.join("scraped_data/final_data", language)):
        makedirs(path.join("scraped_data/final_data", language))

    with open(
        path.join("scraped_data/final_data", language, "data_final.json"),
        "w",
        encoding="utf-8",
    ) as f:
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
    for article in articles:
        if article.get("content"):
            for lang, content in article.get("content").items():
                content["doi"] = article.get("doi")
                languages[lang].append(content)

    return languages


def clean_up_data(data_dir: str):
    articles = json.load(open(path.join(data_dir, "data.json")))

    articles = remove_articles_without_content(data=articles)
    articles = remove_empty_pls(data=articles)
    articles = truncate_articles(data=articles)
    articles = remove_short_abstracts(data=articles)

    languages = get_languages(articles=articles)
    languages = sort_articles_into_languages(articles=articles, languages=languages)

    # TODO Resolve heading etc by index, if length equals english

    # Loop over every language and their articles
    for lang, data in languages.items():
        # split the data into long and sectioned parts
        data_long = [x for x in data if x["pls_type"] == "long"]
        data_sectioned = [x for x in data if x["pls_type"] == "sectioned"]

        # now split long into 1-paragraph and multi-paragraph
        data_long_single = [
            x for x in data_long if len(x["pls"][0].strip().split("\n")) == 1
        ]
        data_long_multi = [
            x for x in data_long if len(x["pls"][0].strip().split("\n")) > 1
        ]

        # truncate all the reviews' plain-language summary appropriately
        for article in data_long_single:
            article["pls"] = one_para_filter(text=article["pls"][0], language=lang)

        for article in data_long_multi:
            first_index = -1
            for index, para in enumerate(article["pls"][0].strip().split("\n")):
                if res_para(text=para, language=lang):
                    first_index = index
                    break

            if first_index > -1:
                article["pls"] = "\n".join(
                    article["pls"][0].strip().split("\n")[first_index:]
                )
            else:
                article["pls"] = ""

        data_long_single = [x for x in data_long_single if len(x["pls"]) > 0]
        data_long_multi = [x for x in data_long_multi if len(x["pls"]) > 0]

        for article in data_sectioned:
            first_index = -1
            for index, section in enumerate(article["pls"]):
                if res_heading(heading=section["heading"], language=lang):
                    first_index = index
                    break

            if first_index > -1:
                article["pls"] = article["pls"][first_index:]
            else:
                article["pls"] = []

        data_sectioned = [x for x in data_sectioned if len(x["pls"]) > 0]

        # now trim based on ratio of pls length to abstract length
        data_long_single = [
            x
            for x in data_long_single
            if (
                pls_length(x) / abs_length(x) >= 0.20
                and pls_length(x) / abs_length(x) <= 1.4
            )
        ]
        data_long_multi = [
            x
            for x in data_long_multi
            if (
                pls_length(x) / abs_length(x) >= 0.30
                and pls_length(x) / abs_length(x) <= 1.3
            )
        ]
        data_sectioned = [
            x
            for x in data_sectioned
            if (
                pls_length(x) / abs_length(x) >= 0.30
                and pls_length(x) / abs_length(x) <= 1.3
            )
        ]
        data_final = data_long_single + data_long_multi + data_sectioned

        save_articles(language=lang, data=data_final)


if __name__ == "__main__":
    data_dir = args.data_dir

    clean_up_data(data_dir=data_dir)
