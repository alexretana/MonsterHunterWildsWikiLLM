# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from  datetime import datetime
from rich.tree import Tree
from rich import print
import pandas as pd
import json

class WikiprojectPipeline:
    def dedupe_and_build_breadcrumb_map(self):
        # read in data, dedupe, rewrite
        filename = './output/fextralife-monsterhunterwildswiki.jsonl'
        with open(filename, 'r', encoding='utf-8'):
            data = [json.loads(line) for line in self.file]
        outputDf = pd.DataFrame(data).drop_duplicates(subset=['url'], keep='last')
        outputDf.to_json(filename, orient='records', lines=True)

        breadCrumbs = outputDf['breadcrumb&title']
        
        tree = Tree("monsterhunterwilds.wiki.fextralife.com:root")
        nodes = {}

        for breadCrumb in breadCrumbs:
            parts = [p for p in path.split('/') if p]
            parent = tree
            partial = ""
            for part in parts:
                partial += "/" + part
                if partial not in nodes:
                    nodes[partial] = parent.add(part)
                parent = nodes[partial]
        
        return tree

    def open_spider(self, spider):
        today = datetime.today().strftime("%Y-%m-%d")
        self.file = open('./output/fextralife-monsterhunterwildswiki.jsonl', 'a', encoding='utf-8')

    def close_spider(self, spider):
        self.file.close()
        breadcrumb_map = self.build_breadcrumb_map()
        with open('./output/fextralife-monsterhunterwildswiki-breadcrumb-map.txt', 'w', encoding='utf-8') as f:
            f.write(breadcrump_map)

    def process_item(self, item, spider):
        line = json.dumps(dict(item)) + "\n"
        self.file.write(line)
        return item
