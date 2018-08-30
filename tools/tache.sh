#!/bin/bash

ssh romeo2 ./envoi_tache.sh 1 $1
RESULT=`./main 32768 $1`
if [ $? = 0 ]
then 
	echo $RESULT >> sorties/$1
	ssh romeo2 ./envoi_tache.sh 2 $1
	rm encours/f$1
else
	rm encours/f$1
	ssh romeo2 ./envoi_tache.sh 3 $1
fi
