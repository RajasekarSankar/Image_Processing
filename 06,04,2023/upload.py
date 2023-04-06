import requests
from pyDataverse.api import NativeApi
from pyDataverse.models import Datafile


class DataverseFile:

    def __init__(self, base_url, api_token, dataverse_alias, dataset_id,filename): #, file_id):
        self.base_url = base_url
        self.api_token = api_token
        self.dataverse_alias = dataverse_alias
        self.dataset_id = dataset_id
        self.api = NativeApi(base_url, api_token)
        self.filename= filename
        #self.file_id = file_id

    
    def upload_file(self, file_path, file_metadata):
        # create a file object

        with open(file_path, 'rb') as file_obj:
            file_name = file_obj.name.split("/")[-1]
            files = {'file': (file_name, file_obj)}

        # upload the file
        upload_url = f"{self.base_url}/datasets/:persistentId/add?persistentId={self.dataset_id}"

        response = requests.post(upload_url, headers={'X-Dataverse-key': self.api_token})

        if response.status_code == 201:
            file_id = response.json()['data']['files'][0]['dataFile']['id']
            print(f"File uploaded successfully with ID {file_id}.")
            # add metadata to the file
            self.api.add_file_metadata(file_id, file_metadata)
        else:
            print("Error uploading file.")
    '''
    def upload_api(self):
        df = Datafile()
        df.set({"pid": self.dataset_id, "filename": self.filename})
        df.get()

        return self.api.upload_datafile(self.dataset_id, self.filename, df.json()), print(self.filename+" has been added succsssfully!!")
    '''
    '''
    def download_file(self):
        # specify the download URL for the file
        download_url = f"{self.base_url}access/datafile/{self.file_id}?key=76b274ab-4eaf-4403-8bb6-ad3929ae3c90"

        # download the file 
        response = requests.get(download_url, verify=False)
        
        with open('16test-1.jpg', 'wb') as file_obj:
            file_obj.write(response.content)
        print("File downloaded successfully.")
    '''
        

