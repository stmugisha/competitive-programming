## Convert image dataset to coco format
import os
import requests

# datasets base class
# dataset type classes


# Download glove vectors
URL = "https://nlp.stanford.edu/data/glove.6B.zip"

def extract_glove_files(file_, delete_zip=False):
	"""Unzip input target file."""
	import zipfile

	#extract zip_file
	zf = zipfile.ZipFile(file_)
	print(f" Extracting {file_} file")
	zf.extractall()

	# Delete original zip
	if delete_zip: 
			print(f" Remove zip {file_+'.zip'} file")
			os.remove(path=zip_file)

def download_glove(url=None, file_='../data/glove.zip'):
	import tqdm
	import requests

	#if the dataset already exists exit
	if os.path.isfile(file_):
			print("dataset already downloded :) ")
			
			return

	print("-"*80)
	print("  Downloading Glove vectors")
	print("-"*80)
	response = requests.get(url, stream=True)
	#read and write chunk by chunk
	handle = open(file_, "wb")
	for chunk in tqdm.tqdm(response.iter_content(chunk_size=512)):
			if chunk:  
					handle.write(chunk)
	handle.close()
	print("Download complete) :")
			
	#extract files
	extract_glove_files(file_)

#download_glove(url=URL)
