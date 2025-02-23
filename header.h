/**
 * \file header.h
 * \brief Define de l'ensemble du programme
 * \author J.Loiseau
 * \version 1.0
 * \date 10/03/2015
 *
 * Détail les données du problème, la répartition en tâches et les informations CPU/GPU
 *
 */

#ifndef HEADER_H_
#define HEADER_H_

/** \addtogroup Define_commun Define commun CPU/GPU
 *  @{
 */
/** @brief Problème traité*/
constexpr int NBCOUL = 27;
/** @brief Nombre de bits pour le nombre de tâches */
constexpr int TAILLETACHE = 33; 


/** @brief Nombre de positions = 2*NBCOUL*/
constexpr int NBCUBE = NBCOUL * 2; 
/** @brief Nombre total de tâches pow(2,TAILLETACHE)*/
constexpr unsigned long long NBTACHE = static_cast<unsigned long long>(1) << TAILLETACHE; 
/** @brief Décalage de la suite de gray = TAILLETACHE + 3*/
constexpr int DECALAGEGRAY = TAILLETACHE + 3; 
/** @}*/

/** \addtogroup Define_HOST Suite de Gray partie HOST 
 *  @{
 */
/** @brief Taille de la suite de G/G en bits pour le CPU (< 20 Mo)*/
constexpr int NBGRAY = 19;				 				/** ATTENTION ! NBGRAY =< 24 */ 
/** @brief pow(2,NBGRAY)-1 */
constexpr unsigned long long TAILLEGRAY = (1 << NBGRAY) - 1; /** ATTENTION ! le -1 */
/** @}*/

/** \addtogroup Define_DEVICE Suite de Gray partie DEVICE
 *  @{
 */
/** @brief Taille de la suite de G/G en bits pour le GPU (< 64KB)*/
constexpr int NBGRAY_GPU = 15; /** ATTENTION ! NBGRAY_GPU =< 15 */ 
/** @brief pow(2,NBGRAY_GPU)-1 */
constexpr unsigned long long TAILLEGRAY_GPU = (1<<NBGRAY_GPU) -1;				/** ATTENTION ! le -1 */
/** @}*/

/** @brief Nombre de threads par blocks sur les GPU*/
constexpr int nbThreads = 128;

#endif /* HEADER_H_ */
