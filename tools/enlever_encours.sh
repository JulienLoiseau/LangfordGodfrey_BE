#!/bin/bash

FILES=encours/*

for f in $FILES 
do 
	nom1=$(basename $f) 
	nom=`echo $nom1 | sed 's/f//g' `
	if [ -e sorties/$nom ]
	then
		rm encours/f$nom
	fi
done
