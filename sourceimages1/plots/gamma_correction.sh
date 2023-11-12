#!/bin/bash

# Loop through each PNG file in the current directory
for file in *.png; do
  # Check if the file is a regular file
  if [ -f "$file" ]; then
    # Apply gamma correction using ImageMagick's convert command
    convert "$file" -gamma 0.4545 "gamma_$file"
  fi
done
