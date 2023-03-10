{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "b624ace1",
   "metadata": {},
   "source": [
    "# Connect to Native API\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "f73b17fb",
   "metadata": {},
   "outputs": [],
   "source": [
    "DATAVERSE = 'http://172.26.120.197:8080/dataverse/keen_demo'\n",
    "BASE_URL = DATAVERSE.split('/dataverse/')[0]\n",
    "API_TOKEN = '76b274ab-4eaf-4403-8bb6-ad3929ae3c90'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "225944e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyDataverse.api import NativeApi\n",
    "api = NativeApi(BASE_URL, API_TOKEN)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "10492ed3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'status': 'OK', 'data': {'version': '5.11.1', 'build': '1069-02e3e92'}}"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resp = api.get_info_version()\n",
    "resp.json()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "8022ad83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "200"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resp.status_code"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2d66a98d",
   "metadata": {},
   "source": [
    "# Create Dataverse Collection¶\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "f9ca8429",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyDataverse.models import Dataverse\n",
    "from pyDataverse.utils import read_file\n",
    "dv = Dataverse()\n",
    "dv_filename = \"dataverse.json\"\n",
    "dv.from_json(read_file(dv_filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6931000e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'alias': 'pyDataverse_fileupload',\n",
       " 'name': 'KEEN_Demonstrator',\n",
       " 'dataverseContacts': [{'contactEmail': 'rajasekar.sankar@tu-dresden.de'}],\n",
       " 'affiliation': 'TU Dresden',\n",
       " 'description': 'Image processing using KEEN demonstrator',\n",
       " 'dataverseType': 'RESEARCH_GROUP'}"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dv.get()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "be6be0cc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'{\\n  \"affiliation\": \"TU Dresden\",\\n  \"alias\": \"pyDataverse_fileupload\",\\n  \"dataverseContacts\": [\\n    {\\n      \"contactEmail\": \"rajasekar.sankar@tu-dresden.de\"\\n    }\\n  ],\\n  \"dataverseType\": \"RESEARCH_GROUP\",\\n  \"description\": \"Image processing using KEEN demonstrator\",\\n  \"name\": \"KEEN_Demonstrator\"\\n}'"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dv.json()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "60803fb8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataverse pyDataverse_fileupload created.\n"
     ]
    }
   ],
   "source": [
    "resp = api.create_dataverse(\":root\", dv.json())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "1526364c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Dataverse pyDataverse_fileupload published.\n"
     ]
    }
   ],
   "source": [
    "resp = api.publish_dataverse(\"pyDataverse_fileupload\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "bec8bf2e",
   "metadata": {},
   "outputs": [],
   "source": [
    "resp = api.get_dataverse(\"pyDataverse_fileupload\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "32c5ff81",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'status': 'OK',\n",
       " 'data': {'id': 89,\n",
       "  'alias': 'pyDataverse_fileupload',\n",
       "  'name': 'KEEN_Demonstrator',\n",
       "  'affiliation': 'TU Dresden',\n",
       "  'dataverseContacts': [{'displayOrder': 0,\n",
       "    'contactEmail': 'rajasekar.sankar@tu-dresden.de'}],\n",
       "  'permissionRoot': True,\n",
       "  'description': 'Image processing using KEEN demonstrator',\n",
       "  'dataverseType': 'RESEARCH_GROUP',\n",
       "  'ownerId': 1,\n",
       "  'creationDate': '2023-03-10T12:26:16Z'}}"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resp.json()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b351dfe2",
   "metadata": {},
   "source": [
    "# Create Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "id": "74f88e62",
   "metadata": {},
   "outputs": [],
   "source": [
    "from pyDataverse.models import Dataset\n",
    "ds = Dataset()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "id": "e73b2579",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds_filename = \"dataset1.json\"\n",
    "ds.from_json(read_file(ds_filename))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 88,
   "id": "19f5b6c3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'citation_displayName': 'Citation Metadata',\n",
       " 'title': 'pyDataverse_fileupload',\n",
       " 'author': [{'authorName': 'Sankar, Rajasekar',\n",
       "   'authorAffiliation': 'TU Dresden'}],\n",
       " 'datasetContact': [{'datasetContactEmail': 'email',\n",
       "   'datasetContactName': 'Sankar, Rajasekar'}],\n",
       " 'dsDescription': [{'dsDescriptionValue': 'Upload/Download of data files using Py_Dataverse interface'}],\n",
       " 'subject': ['Computer and Information Science']}"
      ]
     },
     "execution_count": 88,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ds.get()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 89,
   "id": "d6029cac",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "True"
      ]
     },
     "execution_count": 89,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ds.validate_json()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 90,
   "id": "c58bfc91",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Adding or updating data with set()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 91,
   "id": "84df26f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "ds.get()[\"title\"]\n",
    "ds.set({\"title\": \"pyDataverse_fileupload\"})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 92,
   "id": "4442cef4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#upload the Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 93,
   "id": "b12508f0",
   "metadata": {},
   "outputs": [],
   "source": [
    "resp = api.create_dataset(\"pyDataverse_fileupload\", ds.json())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 94,
   "id": "d3c5b90c",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "{'status': 'ERROR',\n",
       " 'message': 'Validation Failed: Subject is required. (Invalid value:edu.harvard.iq.dataverse.DatasetField[ id=null ]), Author Name is required. (Invalid value:edu.harvard.iq.dataverse.DatasetField[ id=null ]), Description Text is required. (Invalid value:edu.harvard.iq.dataverse.DatasetField[ id=null ]), Contact E-mail is required. (Invalid value:edu.harvard.iq.dataverse.DatasetField[ id=null ]), Title is required. (Invalid value:edu.harvard.iq.dataverse.DatasetField[ id=null ]).java.util.stream.ReferencePipeline$3@7b6a2185'}"
      ]
     },
     "execution_count": 94,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "resp.json()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 95,
   "id": "33073427",
   "metadata": {
    "collapsed": true
   },
   "outputs": [
    {
     "ename": "KeyError",
     "evalue": "'data'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mKeyError\u001b[0m                                  Traceback (most recent call last)",
      "Input \u001b[1;32mIn [95]\u001b[0m, in \u001b[0;36m<cell line: 1>\u001b[1;34m()\u001b[0m\n\u001b[1;32m----> 1\u001b[0m ds_pid \u001b[38;5;241m=\u001b[39m \u001b[43mresp\u001b[49m\u001b[38;5;241;43m.\u001b[39;49m\u001b[43mjson\u001b[49m\u001b[43m(\u001b[49m\u001b[43m)\u001b[49m\u001b[43m[\u001b[49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[38;5;124;43mdata\u001b[39;49m\u001b[38;5;124;43m\"\u001b[39;49m\u001b[43m]\u001b[49m[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpersistentId\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n",
      "\u001b[1;31mKeyError\u001b[0m: 'data'"
     ]
    }
   ],
   "source": [
    "ds_pid = resp.json()[\"data\"][\"persistentId\"]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3392b0c9",
   "metadata": {},
   "outputs": [],
   "source": [
    "resp = api.create_dataset_private_url(ds_pid)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "26172ef8",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d486a8fe",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.12"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
