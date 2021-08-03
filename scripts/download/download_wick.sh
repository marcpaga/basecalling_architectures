#!/bin/bash

links_file=$1
output_dir=$2

while read a b;
do 
	link=$a
	name=$b
	tar_name="${output_dir}/${name}.tar.gz"
	dir_name="${output_dir}/${name}"
	slo_dir="${output_dir}/${name}/sloika_hdf5s"

	if [ -d $dir_name ]
	then
		if [ -d $slo_dir ]
		then
			echo "Deleting sloika files"
        		rm -rv $slo_dir
		else
			echo "Skipping"
		fi
	else
		if [ -f $tar_name ]
		then
			echo "Unpacking data"
        		tar -xvzf $tar_name
			echo "Deleting sloika files"
                        rm -rv $slo_dir
			rm $tar_name
		else
			echo "Downloading: ${name}"
        		wget $link -O $tar_name
			echo "Unpacking data"
                        tar -xvzf $tar_name
                        echo "Deleting sloika files"
                        rm -rv $slo_dir
		fi
	fi
done < $links_file
