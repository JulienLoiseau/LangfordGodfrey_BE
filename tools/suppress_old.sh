#!/bin/bash

FILE=encours/*

for f in $FILE
do 
	nom=$(basename $f)
	ftime=`stat -c %Y $f`
	if [ $ftime -lt $1 ]
	then
		name=`echo $nom | sed 's/f//g'`
                echo $ftime".."$f".."$name
		rm $f
                ssh romeo2 ./envoi_tache.sh 3 $name
                sleep 1
	fi

done

