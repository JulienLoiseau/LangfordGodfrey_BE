/**
 * \file grands_entiers.cu
 * \brief Fonctions principales pour l'utilisation des LONG HOST/DEVICE
 * \author C. Jaillet
 * \version 1.0
 * \date 10/03/2015
 *
 */
#include <stdio.h>
#include <stdlib.h>

#include "general64.h"
#include "grands_entiers64.h"

/**
 * \fn void LONG_cree(LONG *a)
 * \brief Initialise un LONG 
 *
 * \param a pointeur vers le LONG à initialiser
 *
 * Tout est statique, on positionne le nombre de mots et la place par mot à 1 et chaque mot de la séquence à 0 pour l'initialisation
 */
__host__ __device__ void LONG_cree(LONG *a) {
	int i;
	a->nbMots = 1 ;
	/* Tout init à 0 */
	for(i = 0 ; i < NB_MAX_MOTS ; ++i)
	  a->sequence[i] = 0;
	a->placeParMot = 1 ;
	//LONG_raz(a) ;
} /* void LONG_cree(LONG *a) */

/**
 * \fn void LONG_raz(LONG *a)
 * \brief Remise à Zéro d'un LONG
 *
 * \param a pointeur vers le LONG pour la RAZ
 *
 * Comme pour l'initialisation, on positionne le nombre de mots et la place par mot à 1 et chaque mot de la séquence à 0.
 */
__host__ __device__ void LONG_raz(LONG *a) {
  /* deja cree */
  int i ;

  for (i=0 ; i < NB_MAX_MOTS ; i++)
    a->sequence[i] = 0 ;
  a->nbMots = 1 ;
  a->placeParMot = 1 ; /* pour le signe */
} /* void LONG_raz(LONG *a) */

/**
 * \fn void LONG_init_unite(LONG *a, int b)
 * \brief Initialisation d'un LONG avec la valeur b
 *
 * \param a pointeur vers le LONG à initialiser
 * \param b valeur d'initialisation du LONG 
 *
 * Le premier mot de la séquence est initialisé à b après une RAZ. 
 * Le nombre de place par mot augmente de 1.
 */
__host__ __device__ void LONG_init_unite(LONG *a, int b) {
  /* a deja cree */
  /* b est 1 ou -1 */
  LONG_raz(a) ;
  a->sequence[0] = b ;

  a->placeParMot += 1 ;
  /* le signe est gere a part (est amene rechanger n'importe quand) */
} /* void LONG_init_unite(LONG *a, int b) */

/**
 * \fn boolean LONG_estNul(LONG *a)
 * \brief Vérifier si un LONG est nul
 *
 * \param a pointeur vers le LONG à vérifier 
 * \return boolean TRUE si nul, FALSE sinon
 *
 * On vérifie que le nombre de mots est de 1 et que le premier mot est à 0.
 */
__host__ __device__ boolean LONG_estNul(LONG *a) {
  return ( a->nbMots == 1 && a->sequence[0]==0 ) ;
} /* boolean LONG_estNul(LONG *a) */

/**
 * \fn void LONG_ajoute_LONG(LONG *a, LONG *b)
 * \brief Ajouter un LONG (b) dans un LONG (a)
 *
 * \param a LONG, destination de la somme
 * \param b LONG, seconde opérande déjà recalée
 *
 * On ajoute b dans a.
 * A l'arrivée les signes ne sont plus forcéments cohérent, on recale en utilisant LONG_recalePlus (gère le signe)
 */
__host__ __device__ void LONG_ajoute_LONG(LONG *a, LONG *b) {
  int i ;
  if ( a->nbMots < b->nbMots )
    a->nbMots = b->nbMots ;
  for (i=0 ; i < a->nbMots ; i++)
    a->sequence[i] += b->sequence[i] ;

  while ( a->nbMots > 1    &&    a->sequence[ a->nbMots - 1] == 0 )
    a->nbMots -- ;

  a->placeParMot ++ ; /* additif */
  if ( a->placeParMot == TAILLE_MOT ) /* addition ensuite */
    LONG_recale(a) ;
} /* void LONG_ajouter_LONG(LONG *a, LONG *b) */

/**
 * \fn void LONG_multiplie_char(LONG *a, int b)
 * \brief Multiplie le LONG a par l'entier b
 *
 * \param a LONG, destination du produit
 * \param b entier, seconde opérande positive ou négative
 *
 * On multiplie a par b, le produit est enregistré dans le LONG a
 */
__host__ __device__ void LONG_multiplie_char(LONG *a, int b) { /* cumule dans le premier */
  if ( b==0 )
    LONG_raz(a) ;
  else {
    int i ;

    for (i=0 ; i < a->nbMots ; i++)
      a->sequence[i] *= b ;

    if (b<0) b = -b ;
    if (b<32)        a->placeParMot += 5 ;
    else if (b<64)   a->placeParMot += 6 ;
    else /* b<128 */ a->placeParMot += 7 ;
    if ( a->placeParMot >= TAILLE_MOT - 7 ) /* multiplication par un char */
      LONG_recale(a) ;
  }
} /* void LONG_multiplie_char(LONG *a, int b) */

/**
 * \fn void LONG_recale(LONG *a)
 * \brief recale le LONG a, on propage les retenues
 *
 * \param a LONG à recaler
 *
 * A utiliser après les LONG_multiplie_char, et la plupart des LONG_ajoute_LONG
 */
__host__ __device__ void LONG_recale(LONG *a) {
  int i ;
  for (i=0 ; i < NB_MAX_MOTS-1 ; i++) { /* a->nbMots */
    a->sequence[i+1] += ( a->sequence[i] / BASE ) ;
    a->sequence[i] %= BASE ;
  }
  if ( a->nbMots < NB_MAX_MOTS - 1   &&   a->sequence[ a->nbMots] != 0 )
    a->nbMots ++ ;
  else
    while ( a->nbMots > 1   &&   a->sequence[ a->nbMots - 1 ] == 0 )
      a->nbMots -- ;
  a->placeParMot = TAILLE_BASE ;
} /* void LONG_recale(LONG *a) */

/**
 * \fn void LONG_recalePlus(LONG *a)
 * \brief recale le LONG a, on propage les retenues, tient compte du signe
 *
 * \param a LONG à recaler
 *
 * A utiliser après les LONG_multiplie_char, et la plupart des LONG_ajoute_LONG
 */
__host__ __device__ void LONG_recalePlus(LONG *a) { /* recale aussi le signe */
  int i ;
  LONG_recale(a) ;

  if ( a->sequence[a->nbMots - 1] > 0 )
    for ( i = a->nbMots - 2 ; i >= 0 ; i--) {
      if ( a->sequence[i] < 0 ) {
	a->sequence[i] += BASE ;
	a->sequence[i+1] -= 1 ;
      }
    }
  else /* c'est negatif */
    for ( i = a->nbMots - 2 ; i >= 0 ; i--) {
      if ( a->sequence[i] > 0 ) {
	a->sequence[i] -= BASE ;
	a->sequence[i+1] += 1 ;
      }
    }

  while ( a->nbMots > 1    &&    a->sequence[a->nbMots - 1] == 0 )
    a->nbMots -- ;

} /* void LONG_recalePlus(LONG *a) */

/**
 * \fn boolean LONG_modulo(LONG *a, int nbBits)
 * \brief Divise le LONG a par pow(2,nbBits) et met le résultat dans a
 *
 * \param a LONG à recaler
 * \param nbBits puissance de deux de la division
 * \return boolean TRUE si multiple de 2, FALSE autrement
 *
 * A utiliser après les LONG_multiplie_char, et la plupart des LONG_ajoute_LONG
 */
__host__ __device__ boolean LONG_modulo(LONG *a, int nbBits) {
  /* hyp : a est recalePlus et est de quantite positive
   *       nbBits >= 0
   */
  int i, ecartIndex ;
  long diviseur ;

  ecartIndex = nbBits / TAILLE_BASE ;
  nbBits %= TAILLE_BASE ;

  if ( ecartIndex >= a->nbMots )
    return ( LONG_estNul(a) ) ;

  for (i=0 ; i < ecartIndex ; i++)
    if ( a->sequence[i] != 0 )
      return false ;

  /* si on est arrive ici, c'est qu'il n'y a pas eu probleme */

  /* 1 : on modifie a en decalant de ecartIndex mots */
  a->nbMots -= ecartIndex ;
  for (i=0 ; i < a->nbMots ; i++)
    a->sequence[i] = a->sequence[ i + ecartIndex ] ;
  for ( i = a->nbMots ; i < a->nbMots + ecartIndex ; i++ )
    a->sequence[i] = 0;

  /* 2 : on doit verifier qu'il y a suffisamment de '0' a droite */
  diviseur = localPow(2,nbBits) ;
  if ( a->sequence[0] % diviseur != 0 )
    return false ;

  /* 3 : tout s'est bien passe : on va renvoyer 'vrai', mais avant on effectue les dernieres modifications sur a */
  /* 3a : on troite les premeirs mots (le dernier sera a part) */
  for (i=0 ; i < a->nbMots - 1 ; i++) {
    /* on decale le mot courant */
    a->sequence[i] /= diviseur ;
    /* et on recupere la contribution du mot suivant */
    a->sequence[i] += ( (a->sequence[i+1] % diviseur) << (TAILLE_BASE - nbBits) ) ;
  }

  /* 3b : on traite le dernier mot a part */
  a->sequence[ a->nbMots - 1 ] /= diviseur ;
  if ( a->sequence[ a->nbMots - 1 ] == 0 )
    a->nbMots -- ;

  return true ;
} /* boolean LONG_modulo(LONG *a, int nbBits) */


/**
 * \fn void LONG_affiche(LONG *a)
 * \brief Fonction d'affichage d'un LONG, uniquement disponible sur HOST
 *
 * \param a LONG à afficher
 *
 * Affichage de grand entier avec décalage tous les 1000 
 */
void LONG_affiche(LONG *a) { /* a est recalePlus */
  int i ;

  int nb10, j ;
  int * seq10 ;
  long x ;

  /* 0 : allocation */
  nb10 = ( (a->nbMots + 1) * TAILLE_BASE ) / 10 ;
  /* designe ici le nombre MAX de cases en base 1000 :
  * en base 1024, ca serait 1 + (a->nbMots * TAILLE_BASE) / 10 ;
  */
  seq10 = (int *) malloc ( nb10 * sizeof(int) ) ;
  for (j=0 ; j < nb10 ; j++)
    seq10[j] = 0 ;

  /* 1 : a->sequence (base 2^TAILLE_BASE)  vers  seq10 (base 1024=2^10 pour l'instant) */
  for (i=0 ; i < a->nbMots ; i++) { /* i dans a->sequence */
    j = (i * TAILLE_BASE) / 10 ;   /* nb10 dans seq10 */
    x = ( a->sequence[i] << ( (i* TAILLE_BASE) % 10) ) ;
    while (x != 0) {
      seq10[ j++ ] += x % 1024 ;
      x /= 1024 ;
    }
  }
  nb10 = j ; /* ici, nb10 est le nombre de cases utilisees (dans seq10) */

  /* 2 : seq10 de la base 1024 a la base 1000 */
  /* 1024 a + b = 1000 a + (24 a + b) */
  for (i= nb10 - 1 ; i>0 ; i--)
/* i est la premiere case (en partant de la droite) a decaler vers la droite */
    for (j = i ; j < nb10 ; j++)
      seq10[j-1] += (24 * seq10[j]) ;

  /* 3 : recalage sur la base 1000 */
  /* 3a : on recale toutes les cases sauf la derniere pour l'instant */
  for (i=0 ; i < nb10 - 1 ; i++) {
    seq10[i+1] += (seq10[i] / 1000) ;
    seq10[i] %= 1000 ;
  }
  /* 3b : normalisation de la derniere */
  while ( seq10[ nb10 - 1 ] >= 1000 ) {
    seq10[ nb10 ] += (seq10[ nb10 - 1 ] / 1000) ;
    seq10[ nb10 - 1 ] %= 1000 ;
    nb10 ++ ;
  } /* en principe on a pris toutes les precautions pour que ca ne deborde pas */

  /* 4 : et enfin affichage */
  if ( seq10[nb10 - 1] >= 0) {
	  printf("%d ",seq10[ nb10 - 1 ]) ;
	  for (i = nb10 - 2 ; i >= 0 ; i--) {
		if (seq10[i] < 10)       printf("00");
	    else if (seq10[i] < 100) printf("0") ;
		printf("%d ",seq10[i]) ;
	  }
  } else {
	  printf("- %d ", -seq10[ nb10 - 1 ]) ;
	  for (i = nb10 - 2 ; i >= 0 ; i--) {
		if (seq10[i] > -10)       printf("00");
	    else if (seq10[i] > -100) printf("0") ;
		printf("%d ", -seq10[i]) ;
	  }
  }

  /* 5 : liberation de la memoire (allouee localement) */
  free(seq10) ;

} /* void LONG_affiche(LONG *a) */
