"""
This is an example of how to download the contents in a dataset version export archive. For example, if the dataset export contains images, this script will download
the images from the hosted URLs and produce a directory that contains all image files by the dataset splits.

Before running this script, please make sure the following has been completed:
1. A dataset version export (currently only supports protobuf) has been created
2. `CLARIFAI_PAT` should be set as an environment variable
"""

from clarifai.client.dataset import Dataset


def main():
  Dataset().export(save_path='output.zip', local_archive_path='clarifai-data-protobuf.zip')


if __name__ == '__main__':
  main()
