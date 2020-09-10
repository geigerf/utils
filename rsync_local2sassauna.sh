#!/bin/bash

# bash script to sync the local code to the remote sassauna servers

#echo -e "Choose local directory (directory you want to sync to sassauna servers)"
#read localdir
SOURCEDIR=/home/fabian/Documents/Master_thesis/Python_Code/

#echo -e "Choose destination directory (keras or pytorch)"
#read destdir
DESTDIR=/home/msc20f10/Python_Code

rsync -av --exclude-from='exclude_from_sync.txt' $SOURCEDIR msc20f10@sassauna.ee.ethz.ch:$DESTDIR