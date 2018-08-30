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
#define NBCOUL 27
/** @brief Nombre de positions = 2*NBCOUL*/
#define NBCUBE 54
/** @brief Nombre de bits pour le nombre de tâches */
#define TAILLETACHE 33
/** @brief Nombre total de tâches pow(2,TAILLETACHE)*/
#define NBTACHE 8589934592
/** @brief Décalage de la suite de gray = TAILLETACHE + 3*/
#define DECALAGEGRAY 36
/** @}*/

/** \addtogroup Define_HOST Suite de Gray partie HOST 
 *  @{
 */
/** @brief Taille de la suite de G/G en bits pour le CPU (< 20 Mo)*/
#define NBGRAY 19				 				/** ATTENTION ! NBGRAY =< 24 */ 
/** @brief pow(2,NBGRAY)-1 */
#define TAILLEGRAY 524287						/** ATTENTION ! le -1 */
/** @}*/

/** \addtogroup Define_DEVICE Suite de Gray partie DEVICE
 *  @{
 */
/** @brief Taille de la suite de G/G en bits pour le GPU (< 64KB)*/
#define NBGRAY_GPU 15							/** ATTENTION ! NBGRAY_GPU =< 15 */ 
/** @brief pow(2,NBGRAY_GPU)-1 */
#define TAILLEGRAY_GPU 32767				/** ATTENTION ! le -1 */
/** @}*/

/** @brief Nombre de threads par blocks sur les GPU*/
#define nbThreads 128



#endif /* HEADER_H_ */
