#:!/bin/bash

FILES=sorties/*

for f in $FILES
do
	nom=$(basename $f)
	content=`more $f`
	stamp=`stat -c %Y $f`
	echo $nom";"$stamp";"$content  
done 
