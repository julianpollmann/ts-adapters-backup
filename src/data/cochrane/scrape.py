import argparse
from DoiScraper import DoiScraper
from ArticleScraper import ArticleScraper

parser = argparse.ArgumentParser(
    prog="Cochrane Reviews Scraper", description="Scraper for Cochrane reviews"
)
parser.add_argument(
    "type",
    choices=["dois", "articles", "content"],
    help="Which data type should be scraped (dois/articles/content?)",
)
parser.add_argument("--dois", type=str, nargs="+", help="DOIS to scrape")
parser.add_argument(
    "--data_dir", type=str, default="./scraped_data", help="Directory of scraped data"
)
args = parser.parse_args()

if __name__ == "__main__":
    scrape_type = args.type
    data_dir = args.data_dir
    dois = args.dois

    if scrape_type == "dois":
        """Scrape DOIS for all available Reviews"""
        doi_scraper = DoiScraper()
        dois = doi_scraper.scrape()

    if scrape_type == "articles":
        """Scrape Review/Article Metadata , used for extracting different language links.

        If DOIS are passed to the script, these Reviews/Articles will be scraped
        otherwise all dois in dois.txt.
        Each article will be saved in data_dir/json.
        """

        if not dois:
            doi_scraper = DoiScraper()
            dois = doi_scraper.read_from_file()

        article_scraper = ArticleScraper(data_dir=data_dir)
        article_scraper.scrape_articles(dois=dois)

    if scrape_type == "content":
        """Scrape the actual Review/Article content for each language

        If DOIS are passed to the script, these Reviews/Articles will be scraped
        otherwise all all json files withing data_dir will be scraped.
        The scraped data will be saved in data.json
        """

        article_scraper = ArticleScraper(data_dir=data_dir)

        if dois:
            articles = article_scraper.read_json_by_dois(dois=dois)
        else:
            articles = article_scraper.read_json()

        article_scraper.scrape_content(articles=articles)
        article_scraper.save_merged_json()
