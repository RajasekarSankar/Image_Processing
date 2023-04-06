from upload import DataverseFile

# initialize the DataverseFile object
dv_file = DataverseFile("http://172.26.120.197:8080/api", "76b274ab-4eaf-4403-8bb6-ad3929ae3c90", "WHK_RajasekarSankar", "doi:10.5072/FK2/OTJISO","78.jpg") #"137")

file_path = "C:/Users/rajas/OneDrive/Desktop/ScaDS/workfile/Image_Processing-main/new/78.jpg"
file_name = file_path.split("/")[-1]
print(file_name)

# specify the file metadata
file_metadata = {
    "categories": [
        {"name": "Tags", "values": ["Dog images"]},
        {"name": "File Path", "values": ["API_upload-download / Images / Dogs/"]}
    ]
}

# upload a file with metadata
dv_file.upload_file("C:/Users/rajas/OneDrive/Desktop/ScaDS/workfile/Image_Processing-main/new/78.jpg", file_metadata)

# download a file
#dv_file.download_file()
