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
            'wikiproject.pipelines.WikiprojectPipeline': 300,
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

    def parse_breadcrumb(self, response):
        breadcrumb_tags = "/" + "/".join([x for x in response.css('div.breadcrumb-wrapper a::text').getall() if x != '+'])
        return breadcrumb_tags

    def parse_wiki_content(self, response):
        sel = response.selector
        wikicontent = (" ".join([x.strip() for x in sel.xpath('//div[@id="wiki-content-block"]//text()').getall()])).replace('\xa0', ' ')
        return wikicontent

    def parse_wiki_tables(self, response):
        sel = response.selector
        tables = sel.xpath('//table[@class="wiki_table"]').getall()

        normalized_data = []

        for table_html in tables:
            table_sel = Selector(text=table_html)
            
            # Extract headers from <thead> if present, else first <tr>
            headers = []
            thead = table_sel.xpath('./thead')
            if thead:
                headers = thead.xpath('.//th//text()').getall()
                headers = [h.strip() for h in headers if h.strip()]
            else:
                first_tr = table_sel.xpath('.//tr')[0]
                headers = first_tr.xpath('./th//text() | ./td//text()').getall()
                headers = [h.strip() for h in headers if h.strip()]

            # Extract ros, skipping header row if no thead
            if thead:
                rows = table_sel.xpath('./tbody/tr')
            else:
                rows = table_sel.xpath('.//tr')[1:] # skip first header row

            data = []
            max_len = len(headers)

            for row in rows:
                cells = row.xpath('./th | ./td')
                row_data = []
                for cell in cells:
                    # Check for nested table inside cell
                    nested_table = cell.xpath('.//table')
                    if nested_table:
                        nested_html = nested_table.get()
                        nested_html = parser_table_with_selector(nested_html)
                        row_data.append(nested_data)
                    else: 
                        # Prefer alt or title if image present
                        img = cell.xpath('.//img')
                        if img:
                            alt = img.xpath('./@alt').get()
                            title = img.xpath('./@title').get()
                            text = alt or title or cell.xpath('string(.)').get()
                        else:
                            text = cell.xpath('string(.)').get()
                        row_data.append(text.strip() if text else '')

                max_len = max(max_len, len(row_data))
                data.append(row_data)

            # Pad headers or rows to max_len
            if len(headers) < max_len:
                headers += [f"Extra_{i}" for i in range(max_len - len(headers))]

            for r in data:
                r += [''] * (max_len - len(r))
                normalized_data.append(dict(zip(headers, r)))

        return json.dumps(normalized_data,indent=2)

    def make_wiki_doc(self, response):
        title = response.url.split("/")[-1]
        breadcrumb = self.parse_breadcrumb(response)

        wiki_content = self.parse_wiki_content(response)

        table_json_dump = self.parse_wiki_tables(response)
        
        wiki_doc = f"""
            If the user's answer is answered by information in this file, please direct them to {response.url}
            URL: {response.url}
            ####################
            Page Title: {title}
            ####################
            Breadcrumb: {breadcrumb}
            ####################
            Page Content:
            {wiki_content}
            ####################
            Page Tables Stored as JSONs
            {table_json_dump}
        """

        return title, breadcrumb, wiki_doc


    def parse(self, response):
        # Restore state if first page in session
        if self.pages_crawled == 0 and "url_counter" in self.state:
            self.url_counter = Counter(self.state["url_counter"])
            self.pages_crawled = self.state.get("pages_crawled", 0)

        self.pages_crawled += 1

        title, breadcrumb, wiki_doc = self.make_wiki_doc(response)

        yield {
            'url': response.url,
            'title': title,
            'breadcrumb': breadcrumb,
            'breadcrumb&title': breadcrumb + "/" + title,
            'wiki_content': wiki_doc,
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

                self.url_counter[full_url] += 1

                if self.url_counter[full_url] == 1:

                    yield response.follow(href, self.parse)

    def closed(self, reason):
        # Save state
        self.state["url_counter"] = dict(self.url_counter)
        self.state["pages_crawled"] = self.pages_crawled
        # Final estimation
        self.run_estimation()
