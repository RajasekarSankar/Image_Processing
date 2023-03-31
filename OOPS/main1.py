import requests
from pyDataverse.api import NativeApi

class DataverseFile:
    
    def __init__(self, base_url, api_token, dataverse_alias, dataset_id):
        self.base_url = base_url
        self.api_token = api_token
        self.dataverse_alias = dataverse_alias
        self.dataset_id = dataset_id
        self.api = NativeApi(base_url, api_token)
    
    def upload_file(self, file_path):
        # create a file object
        
        with open(file_path, 'rb') as file_obj:
            file_name = file_obj.name.split("/")[-1]
            files = {'file': (file_name, file_obj)}

        # upload the file
        upload_url = f"{self.base_url}datasets/{self.dataset_id}/storageserver"
        response = requests.post(upload_url, headers={'X-Dataverse-key': self.api_token}, files=files)

        if response.status_code == 200:
            print("File uploaded successfully.")
        else:
            print("Error uploading file.")
    
    def download_file(self, file_id):
        # specify the download URL for the file
        download_url = f"{self.base_url}access/datafile/{file_id}?gbrecs=true"

        # download the file
        response = requests.get(download_url)
        if response.status_code == 200:
            file_name = response.headers['Content-Disposition'].split('=')[1]
            with open(file_name, 'wb') as file_obj:
                file_obj.write(response.content)
            print("File downloaded successfully.")
        else:
            print("Error downloading file.")
