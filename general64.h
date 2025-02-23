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
constexpr int ENCOMBREMENT_GPU = 15; /* ie < 64KB */
/** @brief Encombrement maximal de la suite de G/G pour l'HOST **/
constexpr int ENCOMBREMENT = 24; /* ie < 20Mo */
/** @brief Taille des mots mémoire **/
constexpr int TAILLE_MOT = 64; /* pour utiliser l'architecture 64 bits */

__host__ __device__ int localPow(int a, int b) ;

#endif
