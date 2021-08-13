#!/bin/bash

echo "Processing FAB42828"
echo "Untarring and joining fastq"
find "${1}/FAB42828" -name '*.fastq.gz' -exec gzip -d '{}' \;
find "${1}/FAB42828" -name '*.fastq' -exec mv {} "${1}/FAB42828/fastq" \;
cat "${1}/FAB42828/fastq/"*".fastq" > "${1}/FAB42828/fastq/all_basecalls.fasta"
mv "${1}/FAB42828/fastq/all_basecalls.fasta" "${1}/FAB42828/fastq/all_basecalls.fastq"

echo "Spliting fast5"
multi_to_single_fast5 -i "${1}/FAB42828/fast5" -s "${1}/FAB42828/fast5" --recursive -t $2

echo "Merging fast5 and fastq"
tombo preprocess annotate_raw_with_fastqs --fast5-basedir "${1}/FAB42828/fast5" --fastq-filenames "${1}/FAB42828/fastq/all_basecalls.fastq" --overwrite --processes $2

echo "Resquiggleling"
tombo resquiggle "${1}/FAB42828/fast5" "${1}/genomes/homo_sapiens.fasta" --processes $2 --dna --num-most-common-errors 5 --overwrite

echo 'Reporting results'
python annotate_teng.py --fast5-path "${1}/FAB42828/fast5" --output-file "${1}/FAB42828/segmentation_report.txt" --n-cores $2 --report --verbose

echo "Processing FAF04090"
echo "Untarring and joining fastq"
find "${1}/FAF04090" -name '*.fastq.gz' -exec gzip -d '{}' \;
find "${1}/FAF04090" -name '*.fastq' -exec mv {} "${1}/FAF04090/fastq" \;
cat "${1}/FAF04090/fastq/"*".fastq" > "${1}/FAF04090/fastq/all_basecalls.fasta"
mv "${1}/FAF04090/fastq/all_basecalls.fasta" "${1}/FAF04090/fastq/all_basecalls.fastq"

echo "Spliting fast5"
multi_to_single_fast5 -i "${1}/FAF04090/fast5" -s "${1}/FAF04090/fast5" --recursive -t $2

echo "Merging fast5 and fastq"
tombo preprocess annotate_raw_with_fastqs --fast5-basedir "${1}/FAF04090/fast5" --fastq-filenames "${1}/FAF04090/fastq/all_basecalls.fastq" --overwrite --processes $2

echo "Resquiggleling"
tombo resquiggle "${1}/FAF04090/fast5" "${1}/genomes/homo_sapiens.fasta" --processes $2 --dna --num-most-common-errors 5 --overwrite

echo 'Reporting results'
python annotate_teng.py --fast5-path "${1}/FAF04090/fast5" --output-file "${1}/FAF04090/segmentation_report.txt" --n-cores $2 --report --verbose


echo "Processing FAF09968"
echo "Untarring and joining fastq"
find "${1}/FAF09968" -name '*.fastq.gz' -exec gzip -d '{}' \;
find "${1}/FAF09968" -name '*.fastq' -exec mv {} "${1}/FAF09968/fastq" \;
cat "${1}/FAF09968/fastq/"*".fastq" > "${1}/FAF09968/fastq/all_basecalls.fasta"
mv "${1}/FAF09968/fastq/all_basecalls.fasta" "${1}/FAF09968/fastq/all_basecalls.fastq"

echo "Spliting fast5"
multi_to_single_fast5 -i "${1}/FAF09968/fast5" -s "${1}/FAF09968/fast5" --recursive -t $2

echo "Merging fast5 and fastq"
tombo preprocess annotate_raw_with_fastqs --fast5-basedir "${1}/FAF09968/fast5" --fastq-filenames "${1}/FAF09968/fastq/all_basecalls.fastq" --overwrite --processes $2

echo "Resquiggleling"
tombo resquiggle "${1}/FAF09968/fast5" "${1}/genomes/homo_sapiens.fasta" --processes $2 --dna --num-most-common-errors 5 --overwrite

echo 'Reporting results'
python annotate_teng.py --fast5-path "${1}/FAF09968/fast5" --output-file "${1}/FAF09968/segmentation_report.txt" --n-cores $2 --report --verbose



