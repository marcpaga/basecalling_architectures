# Data preparation

```

    |-----------|          |-----------|          |-----------|
    | Wick data |          | Teng data |          | Jain data |
    |-----------|          |-----------|          |-----------|
          |                      |                      |
          |                      |                      |
   download_wich.sh       download_teng.sh       download_jain.sh
          |                      |                      |
          |                      |                      |
   annotate_wich.sh       annotate_teng.sh       annotate_jain.sh
          |                      |                      |
          |                      |                      |
          \                      |                      /
           ----------------------|----------------------
                                 |
                            organize.sh
                                 |
                                 |
                        --------------------
                        |                  |
                        |                  |
                   train_data          test_data
```

## Download 

Download the data using the following scripts:

- `scripts/download/download_wick.sh OUTPUT_DIR_WICK links_wick.txt`: download some of the datasets published in Wick et al., 2019.
- `scripts/download/download_jain.sh OUTPUT_DIR_JAIN`: download one dataset from Jain et al., 2018
- `scripts/download/download_teng.sh OUTPUT_DIR_TENG`: download the datasets from Teng et al., 2019

These scripts download and unpack the data. Tar files are not removed in case something wrong goes with the data processing and we can avoid re-downloading the data. But the tar files are not used again, so they can be deleted at the user discretion.

## Annotate

Annotate the data using the following scripts

- `scripts/annotate/annotate_wick.sh OUTPUT_DIR_WICK NUM_PROCESSES`: based on the references download resquiggles the data using a custom tombo script. The reads from Wick et al. contain a segmentation table, however the segmentation does not seem to correspond correctly with the raw signal and with the reported reference in the fasta file. For this reason we resquiggle the data using the reported references. 

- `scripts/annotate/annotate_teng.sh OUTPUT_DIR_TENG NUM_PROCESSES`: the reads from Teng et al. contain a segmentation table from nanoraw, which was a predecessor tool to Tombo, to keep all resquiggleling the same we resquiggle the data with Tombo.

- `scripts/annotate/annotate_jain.sh OUTPUT_DIR_JAIN NUM_PROCESSES`: 

Data from Teng et al., are already annotated with tombo

## Organize

Organize the data using the following scripts


