#!/bin/bash

echo "Sorties"
sorties=`ls -l sorties/ | wc -l`
echo $sorties
echo "Encours"
encours=`ls -l encours/ | wc -l`
echo $encours
res=$(($sorties*100/32768))
echo "Pourc : "$res 
