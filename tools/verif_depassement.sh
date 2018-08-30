#:!/bin/bash

FILES=sorties/*

for f in $FILES
do
	nom=$(basename $f)
	content=`more $f`
	num=`echo $content | cut -d';' -f6`
	if [ $num -ne 0 ]
	then 
		echo "fichier :$nom"
	fi 
	num=`echo $content | cut -d';' -f5`
	if [ $num -ne 0 ]
	then
		echo "SECOND fichier :$nom"
	fi 
done 
