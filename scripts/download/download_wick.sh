#!/bin/bash

output_dir=$1

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

train_fast5_links="${SCRIPT_DIR}/links_train_wick.txt"
test_fast5_links="${SCRIPT_DIR}/links_test_wick.txt"
references_links="${SCRIPT_DIR}/links_references_wick.txt"
basecalls_links="${SCRIPT_DIR}/links_basecalls_wick.txt"
genomes_links="${SCRIPT_DIR}/links_general_references.txt"

while read a b;
do 
	link=$a
	name=$b
	tar_name="${output_dir}/${name}.tar.gz"
	dir_name="${output_dir}/${name}"

	if [ -d $dir_name ]
	then
		echo  "Skipping: ${name}"
	else
		if [ -f $tar_name ]
		then
			echo "Unpacking data"
        	tar -xvzf $tar_name
		else
			echo "Downloading: ${name}"
        	wget $link -O $tar_name
			echo "Unpacking data"
            tar -xvzf $tar_name
		fi
	fi
done < $train_fast5_links

while read a b;
do 
	link=$a
	name=$b
	tar_name="${output_dir}/${name}.tar.gz"
	dir_name="${output_dir}/${name}"

	if [ -d $dir_name ]
	then
		echo "Skipping: ${name}"
	else
		if [ -f $tar_name ]
		then
			echo "Unpacking data"
        	tar -xvzf $tar_name
		else
			echo "Downloading: ${name}"
        	wget $link -O $tar_name
			echo "Unpacking data"
            tar -xvzf $tar_name
		fi
	fi
done < $test_fast5_links


if [ ! -d "${output_dir}/basecalls_test" ]
then
    mkdir "${output_dir}/basecalls_test"
fi

while read a b;
do
    link=$a
	name=$b
    tar_name="${output_dir}/basecalls_test/${name}/${name}.tar.gz"
    
    if [ ! -d "${output_dir}/basecalls_test/${name}" ]
    then
        mkdir "${output_dir}/basecalls_test/${name}"
    fi
    
    if [ -f "${output_dir}/basecalls_test/${name}/basecalls.fastq" ]
    then
        echo "Skipping: ${name}"
        continue
    fi
    
    if [ -f $tar_name ] 
    then
        tar -xvzf $tar_name -C "${output_dir}/basecalls_test/${name}"
    else
        wget $link -O $tar_name
        tar -xvzf $tar_name -C "${output_dir}/basecalls_test/${name}"
    fi
    
    if [ -f "${output_dir}/basecalls_test/${name}/guppy_v2.1.3-v2.2.3.fastq" ]
    then
        mv "${output_dir}/basecalls_test/${name}/guppy_v2.1.3-v2.2.3.fastq" "${output_dir}/basecalls_test/${name}/basecalls.fastq"
    fi
    
    if [ -f "${output_dir}/basecalls_test/${name}/01_guppy_v2.1.3.fastq" ]
    then
        mv "${output_dir}/basecalls_test/${name}/01_guppy_v2.1.3.fastq" "${output_dir}/basecalls_test/${name}/basecalls.fastq"
    fi
    
done < $basecalls_links


while read a b;
do
    link=$a
	name=$b
    tar_name="${output_dir}/genomes_test/${name}.fna.gz"
    
    if [ -f "${output_dir}/genomes_test/${name}.fna" ]
    then
        echo "Skipping: ${name}"
        continue
    fi
    
    if [ -f $tar_name ] 
    then
        gzip -d $tar_name
    else
        wget $link -O $tar_name
        gzip -d $tar_name
    fi
    
done < $references_links

if [ ! -d "${output_dir}/genomes" ]
then
    mkdir "${output_dir}/genomes"
fi

while read a b;
do
    link=$a
	name=$b
    tar_name="${output_dir}/genomes/${name}.fna.gz"
    
    if [ -f "${output_dir}/genomes_test/${name}.fna" ]
    then
        echo "Skipping: ${name}"
        continue
    fi
    
    if [ -f $tar_name ] 
    then
        gzip -d $tar_name
    else
        wget $link -O $tar_name
        gzip -d $tar_name
    fi
    
done < $genomes_links

