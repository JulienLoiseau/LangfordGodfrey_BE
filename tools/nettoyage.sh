#!/bin/bash

FILE=sorties/*

for f in $FILE
do
	cat $f >> backup_res_24.txt
done
