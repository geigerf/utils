#!/bin/bash

# bash script to sync the local code to the remote sassauna servers

#echo -e "Choose local directory (directory you want to sync to sassauna servers)"
#read localdir
SOURCEDIR=/home/msc20f10/Python_Code/results/

#echo -e "Choose destination directory (keras or pytorch)"
#read destdir
#DESTDIR=/media/sf_Master_thesis/Python_Code/results
DESTDIR=/home/fabian/Documents/results

rsync -av msc20f10@sassauna.ee.ethz.ch:$SOURCEDIR $DESTDIR