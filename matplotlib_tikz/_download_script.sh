#!/bin/bash

mkdir -p /mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz
cd /mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz

while IFS= read -r url; do
  filename=$(basename "$url")
  echo "Downloading $filename from $url"
  wget -O "$filename" "$url"
done < /mnt/ubuntu_hdd/open_source/code/struct_eval/llm/matplotlib_tikz.txt