/**
 * \file langford_God_omp16.cu
 * \brief Corps des fonctions de résolution GPU et fonction principale de résolution hybride
 * \author J.Loiseau
 * \version 1.0
 * \date 10/03/2015
 *
 */

#include <stdlib.h>
#include <stdio.h>
#include <sys/timeb.h>
#include <omp.h>

#include "fonctions.h"

/** \brief Tableau de G/G en mémoire constante du GPU (<64KB) */
__constant__ char d_Gray[TAILLEGRAY_GPU];
/** \brief Début de la tâche (selon le process MPI) */
__constant__ long long d_debutT;
/** \brief Tableau de G/G de l'HOST (<20Mo pour E5-2650v2) */
char h_Gray[TAILLEGRAY];


/**
 * \fn void consGray()
 * \brief Construit la partie en mémoire de la suite de G/G sur l'HOST
 *
 */
void consGray() {
	char valeur ;
	int debut, pas ;
	int i ;

	for ( valeur=0, debut=0, pas=2;	valeur < NBGRAY	; valeur++, debut=2*debut+1, pas*=2 )
		for ( i=debut ; i<TAILLEGRAY ; i+=pas )
			h_Gray[i] = valeur ;
}

/**
 * \fn void d_initLocale(memlocale_gpu *ml)
 * \brief Initialisation de la mémoire locale au thread
 *
 * \param ml mémoire locale au thread
 * Dans notre cas l'on fixe les valeurs des bits 1 et 2 à 1 (Voir preuve)
 */
__device__ void d_initLocale(memlocale_gpu *ml) {
	int i;
	/* on fixe x1 = x2 = 1 */
	for (i=1 ; i<=2 ; i++)
		ml->evaluation[i] = 1 ;
	LONG_cree( &(ml->sommeTache) ) ;
	LONG_cree( &(ml->produit) ) ;
}

/**
 * \fn void d_initTermes(memlocale_gpu *ml)
 * \brief Initialisation des termes par rapport à la tâche actuelle
 *
 * \param ml mémoire locale au thread
 */
__device__ void d_initTermes(memlocale_gpu *ml) {
	int i, j ;
	boolean onDoitMultiplier ;

	/* 1: les premiers termes : ce sont les sommes de produits deux à deux */
	/* 1a : i=1 : x1.x3 + ... + x(n-1).x(n+1) (on s'arrête en cours de route) */
	ml->termes[1] = 0 ;
	for ( j=1 ; j <= NBCOUL - 1 ; j++ )
		ml->termes[1] += ml->evaluation[j] * ml->evaluation[j+2] ;
	if ( NBCOUL % 2 ) { /* S1 paire */
		ml->termes[1] /= 2 ;
		onDoitMultiplier = (ml->termes[1] != 0) ;
	}
	else onDoitMultiplier = true ;

	/* 1b : pour i>1 : x1.x(1+i+1) + x2.x(2+i+1) + ... + x(2n-i-1)x(2n) */
	for (i=2 ; i <= NBCOUL ; i++) {
		ml->termes[i] = 0 ;
		for ( j=1 ; j <= NBCUBE - i - 1 ; j++ )
			ml->termes[i] += ml->evaluation[j] * ml->evaluation[j+i+1] ;
		if ( i%2 ) { /* Si paire */
			ml->termes[i] /= 2 ;
			onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;
		}
		/* else impair : ne peut pas s'annuler */
	}

	/* 2: le signe ici (je compte les negatifs, puis je donne le signe) */
	ml->signe = 0 ;
	for (i=3 ; i < 3 + TAILLETACHE ; i++)
		if (ml->evaluation[i] < 0)
			ml->signe ++ ;
	ml->signe = (ml->signe % 2 == 0) ? 1 : -1 ;

	/* 3: init. de sommeTache est au premier produit (avec le signe) */
	if (onDoitMultiplier) {
		LONG_init_unite( &(ml->sommeTache) , ml->signe) ;
		for (i=1 ; i <= NBCOUL ; i++)
			LONG_multiplie_char( &(ml->sommeTache) , ml->termes[i]) ;
	}
	else
		LONG_raz( &(ml->sommeTache) ) ;
} /* void initTermes(memlocale *ml, memglobale *mg) */


/**
 * \fn void d_initTache(int numero, memlocale_gpu *ml , char * GrayTab)
 * \brief Initialisation de la mémoire locale par rapport à la tâche numero
 *
 * \param numero numéro de la tâche associée
 * \param ml mémoire locale au thread
 * \param GrayTab tableau de gray conservé en mémoire
 *
 * On regarde par rapport à la tâche convertie en binaire quels sont les valeurs des bits dans les positions (evaluation)
 */
__device__ void d_initTache(long long numero, memlocale_gpu *ml , char * GrayTab) {
	long long i ;
	long long v ;

	/* on calcule la tâche, de 3 à 2+NBTACHE */
	v = numero ;
	for ( i=3 ; i < 3 + TAILLETACHE ; i++ ) {
		ml->evaluation[i] = (v%2 == 1) ? -1 : 1 ;
		v /= 2 ;
	}
	/* fin de l'enumeration */
	/* on fixe toutes les valeurs selon Gray à 1 (avant de commencer) */
	for ( i = 3 + TAILLETACHE ; i <= NBCUBE ; i++ )
		ml->evaluation[i] = 1 ;

	/* les termes au debut, et sommeTache init. au produit des termes initiaux */
	d_initTermes(ml) ;

} /* void initTache(int numero, memlocale *ml, memglobale *mg) */

/**
 * \fn void d_modifierCase(int numCase, memlocale_gpu *ml , char * GrayTab)
 * \brief Modification du signe de la case et répercution de la modification
 *
 * \param numCase numéro de la case à modifier dans le tableau evaluation
 * \param ml mémoire locale au thread
 * \param GrayTab tableau de gray conservé en mémoire
 *
 */
__device__ void d_modifierCase(int numCase, memlocale_gpu *ml) {
	int i ;
	boolean onDoitMultiplier = true ;

	int max1, max2 = NBCOUL ;

	ml->evaluation[numCase] *= -1 ;

	/* terme 1 */
	/* x1.x3 + ... + x(n-1).x(n+1) */
	if ( numCase <= NBCOUL - 1 ) /* il y est en tant que membre de gauche ET droite */
	    /* 1 si S1 peut s'annuler ie si n impair (2 sinon) */
		ml->termes[1] += (2 - NBCOUL % 2) * ml->evaluation[numCase] *
							( ml->evaluation[numCase-2] + ml->evaluation[numCase+2] ) ;
	else
	/* y est-il en tant que membre de droite ? */
		/* ok ssi ( numCase >= 3 && numCase <= NBCOUL+1 )
		 * ie ssi ( numCase <= NBCOUL+1 ) car le numCase est une numGray, >= 3
		 *												(les 2 premiers sont fixes)
		 */
		if ( numCase <=NBCOUL + 1 ) /* il y est DONC comme membre de droite */
			ml->termes[1] += (2 - NBCOUL % 2) * ml->evaluation[numCase-2] * ml->evaluation[numCase] ;
	if (NBCOUL % 2)
		onDoitMultiplier = (ml->termes[1] != 0) ;

if (numCase <= NBCOUL) {
	max1 = numCase - 2 ;
	if (numCase >= NBCOUL - 1)
		max2 = 2 * NBCOUL - numCase - 1 ;

	/* TOUS termes */
/* indices pairs : Si ne peut s'annuler */
	for (i=2 ; i <= max1 ; i += 2) {
		/* a la fois membre gauche et membre droit */
		ml->termes[i] += 2 * ml->evaluation[numCase] *
							( ml->evaluation[numCase-i-1] + ml->evaluation[numCase+i+1] ) ;
	}
	for ( ; i <= max2 ; i += 2) {
		/* seulement membre gauche */
		ml->termes[i] += 2 * ml->evaluation[numCase] * ml->evaluation[numCase+i+1] ;
	}
	/* et c'est tout : sinon, il n'est ni membre gauche, ni membre droit */
/* indice impairs : la somme peut s'annuler ; on utilise les moities */
	for (i=3 ; i <= max1 ; i += 2) {
		/* a la fois membre gauche et membre droit */
		ml->termes[i] += ml->evaluation[numCase] *
							( ml->evaluation[numCase-i-1] + ml->evaluation[numCase+i+1] ) ;
		onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;
	}
	for ( ; i <= max2 ; i += 2) {
		/* seulement membre gauche */
		ml->termes[i] += ml->evaluation[numCase] * ml->evaluation[numCase+i+1] ;
		onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;
	}
	/* et c'est tout : sinon, il n'est ni membre gauche, ni membre droit */
	/* mais il se peut encore que le terme d'avant soit nul */
	for ( ; i <= NBCOUL ; i += 2)
		onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;

} else { /* numCase > NBCOUL */
	max1 = 2 * NBCOUL - numCase - 1 ;
	if (numCase <= NBCOUL + 2)
		max2 = numCase - 2 ;
	else if (numCase >= 2 * NBCOUL - 1)
		max1 = 1 ;

	/* voila TOUS LES AUTRES termes */
/* indices pairs : Si ne peut s'annuler */
	for (i=2 ; i <= max1 ; i += 2) {
		/* a la fois membre gauche et membre droit */
		ml->termes[i] += 2 * ml->evaluation[numCase] *	( ml->evaluation[numCase-i-1] + ml->evaluation[numCase+i+1] ) ;
	}
	for ( ; i <= max2 ; i += 2) {
		/* seulement membre droite */
		ml->termes[i] += 2 * ml->evaluation[numCase-i-1] * ml->evaluation[numCase] ;
	}
	/* et c'est tout : sinon, il n'est ni membre gauche, ni membre droit */
/* indice impairs : la somme peut s'annuler ; on utilise les moities */
	for (i=3 ; i <= max1 ; i += 2) {
		/* a la fois membre gauche et membre droit */
		ml->termes[i] += ml->evaluation[numCase] *( ml->evaluation[numCase-i-1] + ml->evaluation[numCase+i+1] ) ;
		onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;
	}
	for ( ; i <= max2 ; i += 2) {
		/* seulement membre droite */
		ml->termes[i] += ml->evaluation[numCase-i-1] * ml->evaluation[numCase] ;
		onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;
	}
	/* et c'est tout : sinon, il n'est ni membre gauche, ni membre droit */
	/* mais il se peut encore que le terme d'avant soit nul */
	for ( ; i <= NBCOUL ; i += 2)
		onDoitMultiplier = onDoitMultiplier && (ml->termes[i] != 0) ;

}

	/* cumul (produit, avec le bon signe ; puis incidence sur le produit) */
	ml->signe = - ml->signe ;
	if ( onDoitMultiplier ) {
		LONG_init_unite( &(ml->produit) , ml->signe) ;
		for (i=1 ; i <= NBCOUL ; i++)
			LONG_multiplie_char( &(ml->produit) , ml->termes[i]) ;

		LONG_recale( &(ml->produit) ) ;
		LONG_ajoute_LONG( &(ml->sommeTache) , &(ml->produit) ) ;
	}

} /* void modifierCase(int numCase, memlocale *ml, memglobale *mg) */

/**
 * \fn void d_accumulerTache(memlocale_gpu *ml, char * GrayTab)
 * \brief Parcours de l'ensemble de la suite de G/G et modification les cases en conséquence
 *
 * \param ml mémoire locale au thread
 * \param GrayTab tableau de gray conservé en mémoire
 *
 */
__device__ void d_accumulerTache(memlocale_gpu *ml, char * GrayTab) {
	int numCase ;
	int j,	/* indice dans le tableau mg->Gray */
	       	 t ; /* les "tours" (voir grand commentaire ci-dessous) */
	long gg ; /* pour le nombre de tours */

	int g, k ;

	g = NBCUBE - 2 - TAILLETACHE - ENCOMBREMENT_GPU ; /* le G-E */
	if (g < 0) g = 0 ;	/* mais en faisant attention */

	gg = localPow(2,g) ; /* le 2^(G-E) qui fait attention */

	/* traiter les cases normales (dont le n° selon Gray est dans le tableau) */
	for ( j = 0 ; j < TAILLEGRAY_GPU ; j++ ) {
		numCase = DECALAGEGRAY + GrayTab[j] ;
		d_modifierCase(numCase, ml) ;
	}

	for (t=1 ; t<gg ; t++) { /*tours suivants */

		/* traiter la dernière du tour precedent */
		/*	 (dont on doit CALCULER à la main le numCase à modifier) */
		k = ENCOMBREMENT_GPU ;
		j = t ;
		while ( j % (TAILLEGRAY_GPU + 1) == 0 ) { /* TAILLEGRAY_GPU = 2^ENCOMBREMENT_GPU */
			k += ENCOMBREMENT_GPU ;
			j /= (TAILLEGRAY_GPU + 1) ;
		}
		numCase = DECALAGEGRAY + GrayTab[j-1] + k ;
		d_modifierCase(numCase, ml) ;

		/* traiter les suivantes du tour : qui sont dans Gray (modulo ...) */
		for ( j = 0 ; j < TAILLEGRAY_GPU ; j++ ) {
			numCase = DECALAGEGRAY + GrayTab[j] ;
			d_modifierCase(numCase,ml) ;
		}

	} /* for (t=1 ; t<gg ; t++) */

} /* void accumulerTache(memlocale *ml, memglobale *mg) */

/**
 * \fn void d_compterTache(int t, memlocale_gpu *ml, char * GrayTab)
 * \brief Métafonction pour l'initialisation et le calcul d'une tâche
 *
 * \param t numéro de la tâche
 * \param ml mémoire locale au thread
 * \param GrayTab tableau de gray conservé en mémoire
 *
 */
__device__ void d_compterTache(long long t, memlocale_gpu *ml, char * GrayTab) {
	/* 1: initialisations */
	d_initTache(t, ml, GrayTab) ;

	/* 2: compter */
	d_accumulerTache(ml, GrayTab) ;
} /* void compterTache(int t, memlocale *ml, memglobale *mg) */

/**
 * \fn void HandleError( cudaError_t err, const char * file, int line)
 * \brief Gestion et affichage des erreurs CUDA pendant l'exécution
 *
 * \param err type de l'erreur
 * \param file fichier dans lequel l'erreur est localisée
 * \param line ligne dans le fichier où est localisée l'erreur
 *
 */
static void HandleError( cudaError_t err, const char * file, int line)
{
	if(err != cudaSuccess)
	{
		printf("%s dans %s en ligne line %d\n",cudaGetErrorString(err),file,line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR(err) (HandleError(err,__FILE__,__LINE__))

/**
 * \fn void resolutionTache(LONG * res)
 * \brief Kernel pour la résolution d'une tâche par un thread
 *
 * \param res tableau des LONG en mémoire DEVICE
 * 
 * Chaque thread s'identifie et calcul sa propre tâche.
 * A la sortie il faut recaler et mettre le LONG résultat dans la case correspondante.
 *
 */
__global__ void resolutionTache(LONG * res)
{
	long long x = blockDim.x * blockIdx.x + threadIdx.x;
	long long tache = x+ d_debutT;
	memlocale_gpu mlT;
	d_initLocale(&mlT);
	d_compterTache(tache,&mlT,d_Gray) ;
	LONG_recale( &(mlT.sommeTache) ) ;
	res[x] = mlT.sommeTache;
}

/**
 * \fn LONG resoudre(int id, int size, int pourcGPU)
 * \brief Fonction principale de résolution hybride CPU/GPU
 *
 * \param id Identifiant MPI du process
 * \param size Nombre total de process MPI
 * \param pourcGPU Pourmille du travail à réaliser par le GPU 
 * 
 * \TODO détailler 
 *
 */
LONG resoudre(int id, int size, int pourcGPU) {


	long long nbTaches = NBTACHE/size;
	long long debutT = id*nbTaches;
	long long finT = debutT + nbTaches;


	//printf("%d in %lu => %lu = (%lu - %lu)\n",id,size,nbTaches,debutT,finT);
	//printf("%d -> %lu taches\n",id,nbTaches);
	
	/* Cas du dernier process MPI, si nombre impair, il fini les tâches */
	if(id == size-1)
	{
		finT = NBTACHE;
	}

	long long i;
	LONG sommeGlobale ;
	LONG_cree(&sommeGlobale) ;
	
/*Constructon de la suite de G/G pour le CPU */
	consGray();
	
	LONG * h_res, * d_res;
	int nbBlocks = 0;
	/********************************** partie GPU ****************************/
if(pourcGPU != 0)
{
	/* Répartition des taches */
	nbBlocks = (int)((nbTaches/nbThreads)*(pourcGPU/1000.f));

	dim3 grid(nbBlocks);
	dim3 blocks(nbThreads);
	//printf("%d GPU : (%d,%d)=%lu, CPU : %lu",id,nbBlocks,nbThreads,nbBlocks*nbThreads,nbTaches-nbBlocks*nbThreads);
	//fflush(stdout);
	
	h_res = (LONG *)malloc(sizeof(LONG)*nbBlocks*nbThreads);

	/* Mémoire device */
	HANDLE_ERROR(cudaMalloc((void**)&d_res,sizeof(LONG)*nbBlocks*nbThreads));
/* On copie uniquement la partie utile G/G pour le GPU */	
	HANDLE_ERROR(cudaMemcpyToSymbol(d_Gray,h_Gray,sizeof(char)*TAILLEGRAY_GPU,0,cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpyToSymbol(d_debutT,&debutT,sizeof(long long),0,cudaMemcpyHostToDevice));	

	/* Résolution GPU */
	resolutionTache <<< grid, blocks >>>(d_res);
}
	/****************************** partie CPU *********************************/
	
if(pourcGPU != 1000)
{
	long long depart = debutT+nbBlocks*nbThreads;

#pragma omp parallel shared(sommeGlobale) default(shared)
	{ /* debut de la zone parallèle */
		//struct timeb tavt, tapt ;
		long long j;
		memlocale ml;
		initLocale(&ml) ;

		/* 1 : chacun travaille de son cote */
#pragma omp for schedule(static)
		for(j=depart ; j < finT ; j++) {
			compterTache(j,&ml,h_Gray) ;
			LONG_recale( &(ml.sommeTache) ) ;
			LONG_ajoute_LONG( &(ml.sommeProc) , &(ml.sommeTache) ) ; /* 2e recalee */
		} /* fin du for parallèle */

		/* 2 : collecte globale des resultats individuels */
		/* 2a : chaque processeur recale sa somme */
		LONG_recale( &(ml.sommeProc) ) ;
		/* 2b : transmission sur la somme gloabale */
		/* attention : accès concurrent */
#pragma omp critical
		LONG_ajoute_LONG( &sommeGlobale, &(ml.sommeProc) ) ;	/* 2e recalee */

	} /* fin de la zone parallèle */
}

if(pourcGPU != 0)
{

	struct timeb tavt, tapt ;
	ftime(&tavt);

	/*******************Récupérer les calculs GPU******************************/
	HANDLE_ERROR(cudaMemcpy(h_res,d_res,sizeof(LONG)*nbBlocks*nbThreads,cudaMemcpyDeviceToHost));
	ftime(&tapt);

	//printf("%d Attente CPU : %.3f\n",id,(double)((tapt.time*1000+tapt.millitm)-(tavt.time*1000+tavt.millitm)) / 1000 ) ;
	fflush(stdout);
	/* Sommer le tout */
	for(i = 0 ; i < nbBlocks*nbThreads ; ++i)
	{
		LONG_ajoute_LONG( &sommeGlobale, &(h_res[i]) ) ;
	}
}
	LONG_recalePlus( &sommeGlobale ) ;
if(pourcGPU != 0)
	HANDLE_ERROR(cudaFree(d_res));
	return sommeGlobale;
} /* main */
