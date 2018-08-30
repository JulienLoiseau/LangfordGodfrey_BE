#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>

#include "general64.h"
#include "grands_entiers64.h"
#include "header.h"

int main(int argc, char * argv[])
{
	if(argc != 2)
	{
		printf("./main [nombreFichiers]\n");
		exit(EXIT_FAILURE);
	}
	
	long nbFiles = atol(argv[1]);
	printf("Lecture de %lu fichiers\n",nbFiles);
	long i = 0;
	LONG sommeGlobale;
	LONG_cree(&sommeGlobale);

	for(i = 0 ; i < nbFiles ; ++i)
	{
		FILE * fic;
		LONG recup;
		LONG_cree(&recup);
		char name[20];
		sprintf(name,"sorties/%lu",i);
		//printf("Fichier : %s\n",name);
		fic = fopen(name,"r");
		float temps;

		/* lecture NB_MAX_MOTS */
		int j;
		//printf("%lu;",i);
		for(j = 0 ; j < NB_MAX_MOTS-1 ; ++j)
		{
			fscanf(fic,"%lu;",&recup.sequence[j]);
			//printf("%lu;",recup.sequence[j]);
		}
		recup.sequence[j] = 0;
		printf("%s:%lu;%lu;%lu;%lu;%lu;%lu;%lu\n",name,recup.sequence[0],recup.sequence[1],recup.sequence[2],recup.sequence[3],recup.sequence[4],recup.sequence[5],recup.sequence[6]);

		fscanf(fic,"%d;%d;%f",&recup.nbMots,&recup.placeParMot,&temps);
		//printf("%d;%d;%f\n",recup.nbMots,recup.placeParMot,temps);
		fclose(fic);
		/*Ajout au total */
		LONG_recale(&recup);
		//printf("recup %lu : ",j);LONG_affiche(&recup);printf("\n");
		LONG_ajoute_LONG(&sommeGlobale,&recup);
	}

	LONG_recalePlus(&sommeGlobale);
		printf("Affichage complet : \n");
	for(i = 0 ; i < NB_MAX_MOTS ; ++i)
	{
		printf("%lu\n",sommeGlobale.sequence[i]);
	}



	printf("Avant modulo \n");LONG_affiche(&sommeGlobale);printf("\n");
	if(!LONG_modulo(&sommeGlobale,(NBCOUL/2)*3-1)){
		printf("P R O B L E M E : ca ne tombe pas juste !!\n\n");
		exit(EXIT_FAILURE);
	}
	LONG_recalePlus(&sommeGlobale);
	printf("\tL(2,%d) = ",NBCOUL);
	LONG_affiche(&sommeGlobale);printf("\n");

	printf("Affichage complet : \n");
	for(i = 0 ; i < NB_MAX_MOTS ; ++i)
	{
		printf("%lu\n",sommeGlobale.sequence[i]);
	}

	return EXIT_SUCCESS;
}
