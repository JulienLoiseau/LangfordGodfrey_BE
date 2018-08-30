#!/bin/bash

#rm log.txt
#rm sorties/*
#rm encours/*
#rm dep.txt 
#rm fin.txt

nbtaches=32768

nbtaches0=32767
nbtaches1=32769
i=0
#echo "Début tâches">>log.txt

#tant que toutes les tâches ne sont pas terminées
while [ `ls -l sorties/ | grep -v ^l | wc -l` != $nbtaches1 ]
do
	#Je regarde les tâches dans l'ordre et si j'en trouve une non lancée et non active je la lance
	for j in `seq 0 $nbtaches0`
	do
		if [ ! -e sorties/$j ]
		then 
			if [ ! -e encours/f$j  ]
			then
				#Si il y a de la place dans la FILE ! < 256 à la fois, de rien de monsieur COSSOU
				NBFile=`squeue -a -u jloiseau | wc -l`
				if [ $NBFile -le 256 ]
				then
					touch encours/f$j
					#echo "Lancement "$j >> log.txt
					#echo "Lancement "$j
					#On lance le job 
					sbatch job.sh $j
				fi
			fi
		fi
	done

	# petite pause de 20 secondes avant de vérifier 
	#sleep 5
done
echo "Fin des tâches">>log.txt
echo "Calcul du total">>log.txt
