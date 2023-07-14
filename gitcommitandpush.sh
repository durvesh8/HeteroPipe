#!/bin/bash

# Ask for input on the commit message
echo "Enter the commit message:"
read commitMessage

# Ask for the file to add to the staging area
echo "Enter the file to add to the staging area:"
read file

# Add the file to the staging area
git add $file

# Commit the changes
git commit -m "$commitMessage"

# Push changes to the repository
git push
