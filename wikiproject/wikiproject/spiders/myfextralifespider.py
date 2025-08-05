import scrapy
from datetime import datetime
import logging
import json
from scrapy.selector import Selector

def safe_list_get(listvar, idx, default):
    try:
        return listvar[idx]
    except IndexError:
        default

class MyFextralifeSpider(scrapy.Spider):
    name = "myfextralifespider"
    allowed_domains = ["monsterhunterwilds.wiki.fextralife.com"]
    start_urls = ["https://monsterhunterwilds.wiki.fextralife.com/Monster+Hunter+Wilds+Wiki"]


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

    def parse_breadcrumb(self, sel):
        breadcrumb_tags = "/" + "/".join([x for x in sel.css('div.breadcrumb-wrapper a::text').getall() if x != '+'])
        second_breadcrumb = str(safe_list_get(x.split('/'), 2, 0))
        return breadcrumb_tags, second_breadcrumb

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

    def make_wiki_doc(self, response, html_node):
        title = response.url.split("/")[-1]
        sel = Selector(text=html_node)

        breadcrumb, second_breadcrumb = self.parse_breadcrumb(sel)

        wiki_content = self.parse_wiki_content(sel)

        table_json_dump = json.dumps(self.parse_wiki_tables(html_node), indent=2)
        
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

        return title, breadcrumb, second_breadcrumb, wiki_doc


    def parse(self, response):
        html_node = response.css('html').get()

        title, breadcrumb, second_breadcrumb, wiki_doc = self.make_wiki_doc(response, html_node)

        doc_filename = f'./output/documents/{(breadcrumb + "-" + title).replace("/","-").strip("-")}.txt'
        with open(doc_filename, 'w', encoding='utf-8') as f:
            f.write(wiki_doc)

        yield {
            'url': response.url,
            'title': title,
            'breadcrumb': breadcrumb,
            'secondbreadcrump': second_breadcrumb, # used specifically for Open WebUI Knowledge grouping
            'breadcrumb&title': breadcrumb + "/" + title,
            'wiki_content': wiki_doc,
            'doc_filepath': doc_filename,
            'updatedAt': datetime.utcnow().isoformat()
        }

        for href in response.css('a::attr(href)').getall():
            if href and self.allowed_domains[0] in href or href.startswith('/'):
                full_url = response.urljoin(href)

                # Skip static asset files
                if full_url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.webp', '.bmp')):
                    continue

                yield response.follow(href, self.parse)
