#!/bin/bash

output_dir=$1

human_link1_fast5="http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF04090-3842965088_Multi_Fast5.tar"
human_link1_fastq="http://s3.amazonaws.com/nanopore-human-wgs/rel6/FASTQTars/FAF04090-3842965088_Multi.tar"
human_link2_fast5="http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAF09968-3439856925_Multi_Fast5.tar"
human_link2_fastq="http://s3.amazonaws.com/nanopore-human-wgs/rel6/FASTQTars/FAF09968-3439856925_Multi.tar"
human_link3_fast5="http://s3.amazonaws.com/nanopore-human-wgs/rel6/MultiFast5Tars/FAB42828-288548394_Multi_Fast5.tar"
human_link3_fastq="http://s3.amazonaws.com/nanopore-human-wgs/rel6/FASTQTars/FAB42828-288548394_Multi.tar"
human_ref_link="https://ftp-trace.ncbi.nlm.nih.gov/ReferenceSamples/giab/release/references/GRCh38/GCA_000001405.15_GRCh38_no_alt_analysis_set.fasta.gz"

wget $human_ref_link -O "${output_dir}/reference.fasta.gz"
gzip -d "${output_dir}/reference.fasta.gz"

wget $human_link1_fast5 -O "${output_dir}/FAF04090_fast5.tar"
wget $human_link1_fastq -O "${output_dir}/FAF04090_fastq.tar"

wget $human_link2_fast5 -O "${output_dir}/FAF09968_fast5.tar"
wget $human_link2_fastq -O "${output_dir}/FAF09968_fastq.tar"

wget $human_link3_fast5 -O "${output_dir}/FAB42828_fast5.tar"
wget $human_link3_fastq -O "${output_dir}/FAB42828_fastq.tar"



