import scrapy
from collections import Counter
from datetime import datetime
import logging

class MyFextralifeSpider(scrapy.Spider):
    name = "myfextralifespider"
    allowed_domains = ["monsterhunterwilds.wiki.fextralife.com"]
    start_urls = ["https://monsterhunterwilds.wiki.fextralife.com/Monster+Hunter+Wilds+Wiki"]
    url_counter = Counter()
    pages_crawled = 0
    ESTIMATION_INTERVAL = 10

    custom_settings = {
        "JOBDIR": f'jobs/daily-fextralife-{datetime.today().strftime("%Y-%m-%d")}',
        "CLOSESPIDER_TIMEOUT": 300,
        "ITEM_PIPELINES": {
            'wikiproject.pipelines.Daily_WikiprojectPipeline': 300,
        },
        "DEPTH_LIMIT": 6,
        "LOG_LEVEL": "INFO",
        "LOG_FILE": f'logs/fextralife-{datetime.today().strftime("%Y-%m-%d")}.log',
        "LOG_STDOUT": True,
        "LOG_ENABLED": True
    }

    def run_estimation(self):
        f1 = sum(1 for c in self.url_counter.values() if c == 1)
        f2 = sum(1 for c in self.url_counter.values() if c == 2)
        u = len(self.url_counter)

        logging.info(f"[Estimation] Unique discovered URLs: {u}")
        logging.info(f"[Estimation] f(singletons): {f1}, f2(doubletons): {f2}")

        if f2 > 0:
            u_total = u + (f1 ** 2) / (2 * f2)
            estimated_left = u_total - u
            estimated_percentage_left = estimated_left / u_total * 100 # %
            logging.info(f"[Estimation] Estimate total unique URLs(Chao1): {u_total:.0f}")
            logging.info(f"[Estimation] Estimate pages left: {estimated_left:.0f} ({estimated_percentage_left:.2f}% of Estimated Total)")

            self.crawler.stats.set_value('estimation/unique_urls', u)
            self.crawler.stats.set_value('estimation/u_total', u_total)
            self.crawler.stats.set_value('estimation/estimated_remaining', estimated_left)
            self.crawler.stats.set_value('estimation/estimated_remaining_percentage', estimated_percentage_left)


        else:
            logging.info("[Estimation] Not enough data to estimate total unique URLs")
            fraction_singletons = f1 / u if u else 0
            if fraction_singletons > 0.5:
                logging.info("[Estimation] Possibly in EARLY phase: still finding lots of new unique URLs")
            else:
                logging.info("[Estimation] Possibly in LATE phase: most discovered URLs are repeated, few are unique")

    def parse(self, response):
        # Restore state if first page in session
        if self.pages_crawled == 0 and "url_counter" in self.state:
            self.url_counter = Counter(self.state["url_counter"])
            self.pages_crawled = self.state.get("pages_crawled", 0)

        self.pages_crawled += 1

        yield {
            'url': response.url,
            'content': response.text
        }
        # Every N pages, run estimate
        if self.pages_crawled % self.ESTIMATION_INTERVAL == 0:
            self.run_estimation()

        for href in response.css('a::attr(href)').getall():
            if href and href.startswith('/'):
                full_url = response.urljoin(href)

                # Skip static asset files
                if full_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp')):
                    continue

                if self.url_counter[full_url] == 0:
                    self.url_counter[full_url] += 1

                    yield response.follow(href, self.parse)

    def closed(self, reason):
        # Save state
        self.state["url_counter"] = dict(self.url_counter)
        self.state["pages_crawled"] = self.pages_crawled
        # Final estimation
        self.run_estimation()
