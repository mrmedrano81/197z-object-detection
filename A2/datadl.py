import requests 
from zipfile import ZipFile


#function for downloading model and for extract_zip_files function
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

#function for extracting dataset from URL
def extract_zip_files(url, filename, target_folder):
    download_url(url, filename)
    with ZipFile(filename, 'r') as zipObj:
        zipObj.extractall(target_folder)


