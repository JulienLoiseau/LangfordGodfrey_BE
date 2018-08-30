/**
 * \file grands_entiers64.h
 * \brief Fonctions pour la gestion des LONG
 * \author J.Loiseau
 * \version 1.0
 * \date 10/03/2015
 *
 */


#ifndef __GRANDS_ENTIERS__
#define __GRANDS_ENTIERS__

/** @brief Taille réelle en bits utilisée pour la représentation (le reste pour les retenues)*/
#define TAILLE_BASE 32
/** @brief Taille réelle utilisable par mots */
#define BASE 4294967296  /* 2^32 = 4,294,967,296 */
/** @brief Nombre de mots dans un LONG */
#define NB_MAX_MOTS 7    /* 96 bits pour n=23, 80 pour 20, 56 pour 19, 49 pour 16
						  * 96 = 3 * 32
					      */

						  
/**
 * \struct LONG
 * \brief Structure pour représenter des entiers signés de plusieurs mots mémoires 64 bits
 */
typedef struct {
  long sequence[NB_MAX_MOTS] ;  /*!< Les mots mémoire utilisés pour stocker les valeurs et les retenues */
  int  nbMots ;      			/*!< Nombre de mots mémoire utilisés */
  int  placeParMot ; 			/*!< Places utilisées par mots */
} LONG ;

/*****************************************************************************/
/* les definitions de constantes (macros) pour utiliser des mots de 64 bits  */
/*****************************************************************************/



/*****************************************************************************/
/*  tout passe par adresse pour eviter les recopies   */
/*****************************************************************************/

__host__ __device__ void LONG_cree(LONG *a) ;

__host__ __device__ void LONG_raz(LONG *a) ;
__host__ __device__ void LONG_init_unite(LONG *a, int b) ;

__host__ __device__ boolean LONG_estNul(LONG *a) ;

__host__ __device__ void LONG_ajoute_LONG(LONG *a, LONG *b) ;     /* ajoute au premier */
__host__ __device__ void LONG_multiplie_char(LONG *a, int b) ;   /* cumule dans le premier */

__host__ __device__ void LONG_recale(LONG *a) ;
__host__ __device__ void LONG_recalePlus(LONG *a) ; /* recale aussi le signe */

__host__ __device__ boolean LONG_modulo(LONG *a, int nbBits) ; /* modifie le premier argument */

void LONG_affiche(LONG *a) ;

/*****************************************************************************/
/*****************************************************************************/

#endif
