import scrapy
from datetime import datetime
import logging
import json
from scrapy.selector import Selector

class MyFextralifeSpider(scrapy.Spider):
    name = "myfextralifespider"
    allowed_domains = ["monsterhunterwilds.wiki.fextralife.com"]
    start_urls = ["https://monsterhunterwilds.wiki.fextralife.com/Monster+Hunter+Wilds+Wiki"]

    # track scheduled unique URLs
    scheduled_urls = set()
    pages_crawled = 0
    ESTIMATION_INTERVAL = 10

    custom_settings = {
        "JOBDIR": f'jobs/daily-fextralife-{datetime.today().strftime("%Y-%m-%d")}',
        "CLOSESPIDER_TIMEOUT": 3600,
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
        pages_scheduled = len(self.scheduled_urls)
        pages_crawled = self.pages_crawled

        if pages_scheduled > 0:
            percent_complete = pages_crawled / pages_scheduled * 100
            estimated_remaining = pages_scheduled - pages_crawled

            logging.info(f"[Estimation] Scheduled unique URLs: {pages_scheduled}")
            logging.info(f"[Estimation] Pages crawled: {pages_crawled}")
            logging.info(f"[Estimation] Remaining: {estimated_remaining} ({percent_complete:.2f}% complete)")


            self.crawler.stats.set_value('estimation/scheduled', pages_scheduled)
            self.crawler.stats.set_value('estimation/crawled', pages_crawled)
            self.crawler.stats.set_value('estimation/remaining', estimated_remaining)
            self.crawler.stats.set_value('estimation/percent_complete', percent_complete)
        else:
            logging.info("[Estimate] No pages scheduled yet.")

    def parse_breadcrumb(self, sel):
        breadcrumb_tags = "/" + "/".join([x for x in sel.css('div.breadcrumb-wrapper a::text').getall() if x != '+'])
        return breadcrumb_tags

    def parse_wiki_content(self, sel):
        wikicontent = (" ".join([x.strip() for x in sel.xpath('//div[@id="wiki-content-block"]//text()').getall()])).replace('\xa0', ' ')
        return wikicontent

    def parse_wiki_tables(self, html):
        sel = Selector(text=html)
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

            # Extract rows, skipping header row if no thead
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
                        nested_data = self.parse_wiki_tables(nested_html)
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

        return normalized_data

    def make_wiki_doc(self, response):
        title = response.url.split("/")[-1]
        sel = response.selector

        breadcrumb = self.parse_breadcrumb(sel)

        wiki_content = self.parse_wiki_content(sel)

        page_html = sel.css('html').get()
        table_json_dump = json.dumps(self.parse_wiki_tables(page_html), indent=2)
        
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
Page Tables Stored as JSON (copy below)
{table_json_dump}
        """

        return title, breadcrumb, wiki_doc


    def parse(self, response):
        # Restore state if first page in session
        if self.pages_crawled == 0 and "scheduled_urls" in self.state:
            self.scheduled_urls = set(self.state["scheduled_urls"])
            self.pages_crawled = self.state.get("pages_crawled", 0)

        self.pages_crawled += 1

        title, breadcrumb, wiki_doc = self.make_wiki_doc(response)

        doc_filename = f'./output/documents/{(breadcrumb + "-" + title).replace("/","-").strip("-")}.txt'
        with open(doc_filename, 'w', encoding='utf-8') as f:
            f.write(wiki_doc)

        yield {
            'url': response.url,
            'title': title,
            'breadcrumb': breadcrumb,
            'breadcrumb&title': breadcrumb + "/" + title,
            'wiki_content': wiki_doc,
            'updatedAt': datetime.utcnow().isoformat()
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

                if full_url not in self.scheduled_urls:
                    self.scheduled_urls.add(full_url)
                    yield response.follow(href, self.parse)

    def closed(self, reason):
        # Save state
        self.state["scheduled_urls"] = list(self.scheduled_urls)
        self.state["pages_crawled"] = self.pages_crawled
        # Final estimation
        self.run_estimation()
