#!/usr/bin/env python3
"""
Dataset Download Script for Multimodal Emotion Recognition

This script provides utilities for downloading the MELD and ESD datasets.
Note: IEMOCAP requires manual request and cannot be automatically downloaded.
"""

import os
import argparse
import requests
import tarfile
import zipfile
import gdown
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Dataset URLs
MELD_URL = "http://web.eecs.umich.edu/~mihalcea/downloads/MELD.Raw.tar.gz"
ESD_GDRIVE_ID = "1scuFwqSbkN7cANxgzMRvJ0Gk9TDWW0if"  # ID for Google Drive link

def download_file(url, destination):
    """
    Download a file from a URL with progress bar
    
    Args:
        url (str): URL to download from
        destination (str): Local path to save the file
    """
    if os.path.exists(destination):
        logger.info(f"File already exists at {destination}. Skipping download.")
        return
    
    logger.info(f"Downloading from {url} to {destination}")
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 Kibibyte
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as file, tqdm(
            desc=os.path.basename(destination),
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def download_from_gdrive(file_id, destination):
    """
    Download a file from Google Drive
    
    Args:
        file_id (str): Google Drive file ID
        destination (str): Local path to save the file
    """
    if os.path.exists(destination):
        logger.info(f"File already exists at {destination}. Skipping download.")
        return
    
    logger.info(f"Downloading from Google Drive (ID: {file_id}) to {destination}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    gdown.download(id=file_id, output=destination, quiet=False)

def extract_archive(archive_path, extract_path):
    """
    Extract a tar.gz or zip archive
    
    Args:
        archive_path (str): Path to the archive file
        extract_path (str): Directory to extract to
    """
    logger.info(f"Extracting {archive_path} to {extract_path}")
    os.makedirs(extract_path, exist_ok=True)
    
    if archive_path.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(path=extract_path)
    elif archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
    else:
        logger.error(f"Unsupported archive format: {archive_path}")
        raise ValueError(f"Unsupported archive format: {archive_path}")

def download_meld(data_dir):
    """
    Download and extract the MELD dataset
    
    Args:
        data_dir (str): Base data directory
    """
    meld_dir = os.path.join(data_dir, 'meld', 'raw')
    archive_path = os.path.join(meld_dir, 'MELD.Raw.tar.gz')
    
    logger.info("Downloading MELD dataset...")
    download_file(MELD_URL, archive_path)
    
    logger.info("Extracting MELD dataset...")
    extract_archive(archive_path, meld_dir)
    
    logger.info("MELD dataset downloaded and extracted successfully.")

def download_esd(data_dir):
    """
    Download and extract the ESD dataset
    
    Args:
        data_dir (str): Base data directory
    """
    esd_dir = os.path.join(data_dir, 'esd', 'raw')
    archive_path = os.path.join(esd_dir, 'ESD.zip')
    
    logger.info("Downloading ESD dataset...")
    download_from_gdrive(ESD_GDRIVE_ID, archive_path)
    
    logger.info("Extracting ESD dataset...")
    extract_archive(archive_path, esd_dir)
    
    logger.info("ESD dataset downloaded and extracted successfully.")

def iemocap_instructions():
    """
    Print instructions for obtaining the IEMOCAP dataset
    """
    logger.info("=" * 80)
    logger.info("IEMOCAP DATASET ACCESS INSTRUCTIONS")
    logger.info("=" * 80)
    logger.info("The IEMOCAP dataset requires manual request and cannot be automatically downloaded.")
    logger.info("To obtain access:")
    logger.info("1. Visit: https://sail.usc.edu/iemocap/iemocap_release.htm")
    logger.info("2. Fill out the electronic release form")
    logger.info("3. Wait for approval and download instructions")
    logger.info("4. Once downloaded, place the data in: data/iemocap/raw/")
    logger.info("=" * 80)

def main():
    parser = argparse.ArgumentParser(description='Download datasets for multimodal emotion recognition')
    parser.add_argument('--data_dir', type=str, default='data', help='Base data directory')
    parser.add_argument('--datasets', nargs='+', choices=['meld', 'esd', 'all'], default=['all'],
                        help='Datasets to download (meld, esd, or all)')
    
    args = parser.parse_args()
    data_dir = os.path.abspath(args.data_dir)
    
    # Print IEMOCAP instructions regardless of selection
    iemocap_instructions()
    
    datasets = args.datasets
    if 'all' in datasets:
        datasets = ['meld', 'esd']
    
    for dataset in datasets:
        if dataset == 'meld':
            download_meld(data_dir)
        elif dataset == 'esd':
            download_esd(data_dir)

if __name__ == "__main__":
    main()
