# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from  datetime import datetime
from rich.console import Console
from rich.tree import Tree
from rich import print
import pandas as pd
import json
import os

class WikiprojectPipeline:
    def dedupe_and_build_breadcrumb_map(self):
        # read in data, dedupe, rewrite
        filename = './output/fextralife-monsterhunterwildswiki.jsonl'
        with open(filename, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        outputDf = pd.DataFrame(data).drop_duplicates(subset=['url'], keep='last')
        outputDf.to_json(filename, orient='records', lines=True)

        breadCrumbs = outputDf['breadcrumb&title']
        
        tree = Tree("monsterhunterwilds.wiki.fextralife.com:root")
        nodes = {}

        for breadCrumb in breadCrumbs:
            parts = [p for p in breadCrumb.split('/') if p]
            parent = tree
            partial = ""
            for part in parts:
                partial += "/" + part
                if partial not in nodes:
                    nodes[partial] = parent.add(part)
                parent = nodes[partial]
        
        total_page_count = len(outputDf)
        return tree, total_page_count

    def open_spider(self, spider):
        today = datetime.today().strftime("%Y-%m-%d")
        self.file = open('./output/fextralife-monsterhunterwildswiki.jsonl', 'a', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

        jobdir = spider.settings.get('JOBDIR')
        active_requests_file = os.path.join(jobdir, 'requests.queue', 'active.json')

        remaining_requests = 0
        if os.path.exists(active_requests_file):
            with open(active_requests_file, 'r', encoding='utf-8') as f:
                try:
                    queue_data = json.load(f)
                    remaining_requests = len(queue_data)
                except json.JSONDecodeError:
                    remaining_requests = 0

        breadcrumb_map, total_page_count = self.dedupe_and_build_breadcrumb_map()
        console = Console(record=True)
        console.print(breadcrumb_map)
        output = console.export_text()

        with open('./output/fextralife-monsterhunterwildswiki-breadcrumb-map.txt', 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"Total Pages Already Scraped And Stored: {total_page_count}")
        print(f"Queued Requests Leftover After Early Stop: {remaining_requests}")

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
