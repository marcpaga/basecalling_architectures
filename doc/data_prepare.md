# Data preparation

## Download 

Download the data using the following scripts:

- `scripts/download/download_wick.sh OUTPUT_DIR links_wick.txt`: download some of the datasets published in Wick et al., 2019.
- `scripts/download/download_human.sh`: download one dataset from Jain et al., 2018
- `scripts/download/download_teng.sh OUTPUT_DIR`: download the datasets from Teng et al., 2019

These scripts download and unpack the data. Tar files are not removed in case something wrong goes with the data processing and we can avoid re-downloading the data. But the tar files are not used again, so they can be deleted at the user discretion.

## Annotate

Annotate the data using the following scripts

- `scripts/annotate/annotate_wick.sh OUTPUT_DIR_WICK NUM_PROCESSES`: based on the references download resquiggles the data using a custom tombo script. The reads from Wick et al. contain a segmentation table, however the segmentation does not seem to correspond correctly with the raw signal and with the reported reference in the fasta file. For this reason we resquiggle the data using the reported references. 

## Organize

Organize the data using the following scripts


