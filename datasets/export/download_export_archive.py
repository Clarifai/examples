"""
This is an example of how to download the contents in a dataset version export archive. For example, if the dataset export contains images, this script will download
the images from the hosted URLs and produce a directory that contains all image files by the dataset splits.

Before running this script, please make sure the following has been completed:
1. A dataset version export (currently only supports protobuf) has been created
2. An API KEY for the app has been generated.
"""

import requests

from clarifai.datasets.export.dataset_inputs import DatasetExportReader, InputDownloader

API_KEY = "" # Fill in your API_KEY
local_archive_path = "" # Fill in the full path to the zip file
save_path = "output.zip"

def main():
    metadata = f'Key {API_KEY}'
    # Create a session object and set auth header
    session = requests.Session()
    session.headers.update({'Authorization': metadata})

    # If the dataset export is created via API, the URL can be passed into DatasetExportReader (instead of the local archive path) via the `archive_url` parameter.
    with DatasetExportReader(session=session, local_archive_path=local_archive_path) as reader:
        InputDownloader(session, reader).download_input_archive(save_path=save_path)


if __name__ == '__main__':
    main()