import pandas as pd
import requests
import json
import os

OPEN_WEBUI_DOMAIN_NAME = 'http://localhost'
KNOWLEDGE_LIST = ['Weapons', 'Armor', 'Items', 'Decorations', 'Misc']
_API_KEY = None

DOCUMENTS_DIR = "./output/documents"

def read_dot_api_key():
    global _API_KEY
    if _API_KEY is None:
        # Assuming your API key is in a file named api_key.txt
        try:
            with open('.open_webui_api_key', 'r') as f:
                _API_KEY = f.read().strip()
            if _API_KEY is None:
                raise ValueError("API key not found in file or environment.")
        except FileNotFoundError:
            raise FileNotFoundError("'.open_webui_api_key' file not found.")
    return _API_KEY

def make_headers(extra_headers: dict = None):
    api_key = read_dot_api_key()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if extra_headers:
        headers.update(extra_headers)
    return headers

def print_response_error(response):
    error_message = f"Recieved Non-Successful Status Code({response.status_code}), and message :{response.text}"
    print(error_message)
    return error_message
    
def get_remote_files():
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/files/"
    api_key = read_dot_api_key()
    headers = make_headers()
    response = requests.get(url=full_url, headers=headers)
    if response.status_code not in range(200, 299):
        return print_response_error(response)
    response_json = response.json()
    if not response_json:
        return {}
    return {file["filename"]: file["id"] for file in response_json}

def get_knowledge_list():
    # Check if collections already exists
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/list"
    api_key = read_dot_api_key()
    headers = make_headers()
    response = requests.get(url=full_url, headers=headers)
    # Exit early if api call fails
    if response.status_code not in range(200,299):
        return print_response_error(response)

    # Parse response for list of exiting knowledges. See which are missing
    response_json = response.json()
    return response_json

def upload_file(filepath, knowledge_id):
    # First Upload File to Open Web UI
    full_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/files/"
    headers = make_headers()
    files = {"file": open(filepath, 'rb')}
    upload_response = requests.post(url=full_url, headers=headers, files=files)
    if upload_response.status_code not in range(200, 299):
        print_response_error(upload_response)
        return ""

    upload_response_json = upload_response.json()
    file_upload_name = upload_response_json['name']
    file_id = upload_response_json["id"]
    print(f"Uploaded file: '{file_upload_name}'")

    # Second Add File to knowledge
    add_to_knowledge_full_url = OPEN_WEBUI_DOMAIN_NAME + f":8080/api/v1/knowledge/{knowledge_id}/file/add"
    data = {'file_id': file_id}
    add_to_knowledge_response = requests.post(url=add_to_knowledge_full_url, headers=headers, json=data)
    if add_to_knowledge_response.status_code not in range(200, 299):
        return print_response_error(add_to_knowledge_response)
    
    print(f"Added file {file_upload_name}(file_id: {file_id}) to knowledge at ({knowledge_id})")
    return file_id

def update_file_content(filepath, file_id):
    with open(filepath, 'rb') as f:
        content = f.read()
    full_url = OPEN_WEBUI_DOMAIN_NAME + f":8080/api/v1/files/{str(file_id)}/data/content/update"
    headers = make_headers()
    data = {"content": content.decode('utf-8')}
    update_content_response = requests.post(url=full_url, headers=headers, json=data)
    if update_content_response.status_code not in range(200, 299):
        return print_response_error(update_content_response)

    print(f"Update_content_response")


def upload_or_update_files(df, remote_files):
    """
    note: df should be outputDf read from .jsonl, 
    and this will return outputDf with file_ids
    """
    knowledge_ids = { knowledge['name']: knowledge['id'] for knowledge in get_knowledge_list().items() }

    for idx, row in df.iterrows():
        filepath = row["doc_filepath"]
        filename = os.path.basename(filepath)
        knowledge_name = row.get("secondbreadcrumb", "Misc")
        knowledge_name = knowledge_name if knowledge_name in KNOWLEDGE_LIST else "Misc"
        knowledge_id = knowledge_ids[knowledge_name]

        if filename not in remote_files:
            print(f"Uploading new file: {filename} to {knowledge_name}({knowledge_id})")
            file_id = upload_file(filepath, knowledge_id)
            df.at[idx, "remote_file_id"] = file_id
        else:
            file_id = remote_files[filename]
            print(f"Updating existing file: {filename}({file_id}) in {knowledge_name}({knowledge_id})")
            update_file_content(filepath, file_id)
            df.at[idx, "remote_file_id"] = file_id

    return df

def create_all_collections():
    response_json = get_knowledge_list()
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
        print(f"Attempting to create knowledge: {missing_knowledge}")
        create_knowledge_url = OPEN_WEBUI_DOMAIN_NAME + ":8080/api/v1/knowledge/create"
        headers = make_headers()
        data = {
            "name": missing_knowledge,
            "description": f"Create fextralife's '{missing_knowledge}' knowledge partition",
            "access_control": {
                "public": True,
            },
        }
        create_response = requests.post(url=create_knowledge_url, json=data, headers=headers)
        if create_response.status_code not in range(200,299):
            return print_response_error(create_response)

        print(f"Creation Succeeded for knowledge: {missing_knowledge}")
        print(f"Confirmation response: {json.dumps(create_response.json(), indent=2)}")

def dedupe_and_build_breadcrumb_map():
    # read in data, dedupe, rewrite
    filename = './output/fextralife-monsterhunterwildswiki.jsonl'
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    outputDf = pd.DataFrame(data).drop_duplicates(subset=['url'], keep='last')

    # Deal with uploading files to openwebui, adds file_id to outputDf
    remote_files = get_remote_files()
    outputDf = upload_or_update_files(outputDf, remote_files)

    # Rewrite after deduping and adding file_ids
    outputDf.to_json(filename, orient='records', lines=True)

    # Build tree and total page count
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