# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from  datetime import datetime
import json

class Daily_WikiprojectPipeline:
    def open_spider(self, spider):
        today = datetime.today().strftime("%Y-%m-%d")
        self.file = open(f'./output/daily-fextralife-{today}.jsonl', 'a', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
