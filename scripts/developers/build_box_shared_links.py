import getpass
import json
from pathlib import Path

from boxsdk import Client, OAuth2

# these all come from the box developer app
CLIENT_ID = input("Client ID:")
CLIENT_SECRET = getpass.getpass("Client Secret:")
ACCESS_TOKEN = getpass.getpass("Access Token:")

# pull this from the box url
FOLDER_ID = "132339394972"

# where to write the model links
OUTDIR = Path("../powertrain/resources/default_models/external_model_links.json")

oauth2 = OAuth2(CLIENT_ID, CLIENT_SECRET, access_token=ACCESS_TOKEN)
client = Client(oauth2)

folder = client.folder(folder_id=FOLDER_ID)

files = folder.get_items()

download_links = {}
for f in files:
    name = f.name.split(".")[0]
    download_links[name] = f.get_shared_link_download_url(access='open')

with open(OUTDIR, 'w', encoding='utf-8') as f:
    json.dump(download_links, f, ensure_ascii=False, indent=4)
