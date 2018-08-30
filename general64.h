/**
 * \file general64.h
 * \brief Encombrement et la représentation 64bits
 * \author J.Loiseau
 * \version 1.0
 * \date 10/03/2015
 *
 */

#ifndef __GENERAL__
#define __GENERAL__

/** @brief Encombrement maximal de la suite de G/G pour le DEVICE **/
#define ENCOMBREMENT_GPU 15		/* ie < 64KB */
/** @brief Encombrement maximal de la suite de G/G pour l'HOST **/
#define ENCOMBREMENT 24			/* ie < 20Mo */
/** @brief Taille des mots mémoire **/
#define TAILLE_MOT 64     /* pour utiliser l'architecture 64 bits */

/** @brief Utilisation de boolean = entier **/
#define boolean int
/** @brief boolean FALSE = 0 **/
#define false 0
/** @brief boolean FALSE = non(0) **/
#define true !0

__host__ __device__ int localPow(int a, int b) ;

#endif
