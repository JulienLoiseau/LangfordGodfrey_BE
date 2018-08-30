#!/bin/bash

output="fichier_bk_taches.txt"

for id in {0..32767} 
do
	echo -n $id";" >> $output
	cat sorties/$id >> $output
done

