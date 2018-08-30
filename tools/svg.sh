#!/bin/bash

FILES=sorties/*

for f in $FILES
do
	more $f >> sorties_27.txt

done
