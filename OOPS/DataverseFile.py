from main1 import DataverseFile

# initialize the DataverseFile object
dv_file = DataverseFile("http://172.26.120.197:8080", "76b274ab-4eaf-4403-8bb6-ad3929ae3c90", "WHK_RajasekarSankar", "doi:10.5072/FK2/OTJISO")

# upload a file
dv_file.upload_file("C:/Users/rajas/OneDrive/Desktop/ScaDS/workfile/Image_Processing-main/new/16.jpg")

# download a file
dv_file.download_file("111")