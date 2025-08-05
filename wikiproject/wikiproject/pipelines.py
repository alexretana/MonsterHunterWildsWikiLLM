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
import requests
import json
import os

OPEN_WEBUI_DOMAIN_NAME = 'http://localhost'
KNOWLEDGE_LIST = ['Weapons', 'Armor', 'Items', 'Decorations', 'Misc']


def read_dot_api_key():
    with open('.open_webui_api_key', 'r') as f:
        open_webui_api_key = f.read()
    return open_webui_api_key

def create_all_collections():
    # Check if collections already exists
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/list"
    api_key = read_dot_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    response = requests.get(url=full_url, headers=headers)
    # Exit early if api call fails
    if response.status_code not in range(200,299):
        error_message = f"Recieved Non-Successful Status Code({response.status_code}), and message :{response.text}"
        print(error_message)
        return error_message

    # Parse response for list of exiting knowledges. See which are missing
    response_json = response.json()
    confirmed_knowledges = []
    for knowledge in response_json:
        confirmed_knowledges.append(knowledge["name"])
    missing_knowledges = list(set(KNOWLEDGE_LIST) - set(confirmed_knowledges))
    print(f"Aleady existing knowledges: {confirmed_knowledges}")
    if missing_knowledges:
        print(f"Missing knowledges to create: {', '.join(missing_knowledges)}")
    else:
        print("There are no knowledges to create")
        return

    # Send Create Knowledge API call for each missing knowledge
    for missing_knowledge in missing_knowledges:
        create_knowledge_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/create"
        data = {
            "name": missing_knowledge,
            "description": f"Create fextralife's '{missing_knowledge}' knowledge partition",
            "access_control": {
                "public": True,
            },
        }
        print(f"Attempting to create knowledge: {missing_knowledge}")
        create_response = requests.post(url=create_knowledge_url, json=data, headers=headers)
        if create_response.status_code not in range(200,299):
            error_message = f"Recieved Non-Successful Status Code({create_response.status_code}), and message :{create_response.text}"
            print(error_message)
            return error_message

        print(f"Creation Succeeded for knowledge: {missing_knowledge}")
        print(f"Confirmation response: {json.dumps(create_response.json(), indent=2)}")

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
        create_all_collections()

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
