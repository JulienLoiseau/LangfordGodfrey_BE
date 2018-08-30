/**
 * \file general.cu
 * \brief Routines spéciales l'ensemble du programme
 * \author C. Jaillet
 * \version 1.0
 * \date 10/03/2015
 *
 */

#include "general64.h"

/* juste les routines ici */

/**
 * \fn localPow(int a, int b)
 * \brief Fonction de puissance HOST et DEVICE
 *
 * \param a Nombre
 * \param b Exposant
 * \return r Valeur a puissance b
 */
__host__ __device__ int localPow(int a, int b) {
  int i ;
  int r=1 ;
  for(i=0 ; i<b ; i++)
    r*=a ;
  return r ;
} /* int localPow(int a, int b) */

