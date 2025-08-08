# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

# import local utils
from .utils import *

# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from  datetime import datetime
from rich.console import Console
from rich.tree import Tree
from rich import print
import pandas as pd
import requests
import json
import os

OPEN_WEBUI_DOMAIN_NAME = 'http://localhost'


class WikiprojectPipeline:
    def open_spider(self, spider):
        today = datetime.today().strftime("%Y-%m-%d")
        self.file = open('./output/fextralife-monsterhunterwildswiki.jsonl', 'a', encoding='utf-8')
        
        # Optionally purge OpenWebUI files once (can be enabled/disabled as needed)
        # purge_openwebui()

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

        breadcrumb_map, total_page_count = dedupe_and_build_breadcrumb_map()
        console = Console(record=True)
        console.print(breadcrumb_map)
        output = console.export_text()

        with open('./output/fextralife-monsterhunterwildswiki-breadcrumb-map.txt', 'w', encoding='utf-8') as f:
            f.write(output)

        print(f"Total Pages Already Scraped And Stored: {total_page_count}")
        print(f"Queued Requests Leftover After Early Stop: {remaining_requests}")
        print("Data has been ingested into Chroma vector store via dedupe_and_build_breadcrumb_map()")

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
