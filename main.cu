/**
 * \file main.cu
 * \brief Code process MPI
 * \author J.Loiseau
 * \version 1.0
 * \date 10/03/2015
 *
 * Code pour un process MPI
 *
 */

#include "main.h"

/**
 * \fn int main (int argc, char * argv[])
 * \brief Entrée du programme, chaque process MPI
 *
 * \param argc Nombre d'arguments (= 2 avec la partie GPU en pourmilles )
 * \param argv Arguments du programme
 * \return 0 - Arrêt normal du programme.
 */
int main(int argc, char * argv[])
{

	int world_size, rank;

	world_size = atoi(argv[1]);
	rank = atoi(argv[2]);

	struct timeb tav, tap ;	
	double te;
	int repartition = 600;
	
	ftime(&tav);
	
	LONG sous_total;
	LONG_cree(&sous_total);
	sous_total = resoudre(rank,world_size,repartition);

	ftime(&tap);

	te = (double)((tap.time*1000+tap.millitm)-(tav.time*1000+tav.millitm))/1000 ;

	int i;
	for(i = 0 ; i < NB_MAX_MOTS ; i++)
	{
		printf("%lu;",sous_total.sequence[i]);
	}
	printf("%d;%d;%.4f",sous_total.nbMots,sous_total.placeParMot,te);
		
	return 0;
}

