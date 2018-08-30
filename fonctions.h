/**
 * \file fonctions.h
 * \brief Résolution Godfrey device et host
 * \author J.Loiseau
 * \version 1.0
 * \date 10/03/2015
 *
 * Description des structures pour le device/host
 * Ensemble des fonctions Device/Host pour la résolution d'une tâche 
 *
 */

#ifndef __FUNCTIONS__
#define __FUNCTIONS__

#include "general64.h"
#include "grands_entiers64.h"
#include "header.h"

/**
 * \struct memlocale
 * \brief Mémoire locale à chacun des threads CPU ou GPU
 *
 */
typedef struct {
	int signe;				/*!< Signe actuel, initialisé selon la tâche */
	int evaluation[NBCUBE+1] ;	/*!< Chaque case correpond à une position = (-1;1) */
	int termes[NBCOUL+1] ;		/*!< Valeur de chaque ligne */
	LONG produit;				/*!< LONG pour réaliser le produit des termes */
	LONG sommeTache ;			/*!< LONG pour sommer les produits à chaque tour */	
	LONG sommeProc ;			/*!< Somme totale du coeur pour l'ensemble des tâches */
} memlocale ;

/**
 * \struct memlocale_gpu
 * \brief Mémoire locale à chacun des threads CPU ou GPU
 *
 */
typedef struct {
	int signe;				/*!< Signe actuel, initialisé selon la tâche */
	int evaluation[NBCUBE+1] ;	/*!< Chaque case correpond à une position = (-1;1) */
	int termes[NBCOUL+1] ;		/*!< Valeur de chaque ligne */
	LONG produit;				/*!< LONG pour réaliser le produit des termes */
	LONG sommeTache ;			/*!< LONG pour sommer les produits à chaque tour */	
} memlocale_gpu ;

/* Fonctions de résolution HOST */
void consGray();
void initLocale(memlocale *ml);
void initTermes(memlocale *ml);
void initTache(long long numero, memlocale *ml , char * GrayTab);
void modifierCase(int numCase, memlocale *ml);
void accumulerTache(memlocale *ml, char * GrayTab);
void compterTache(long long t, memlocale *ml, char * GrayTab);

/* Fonctions de résolution DEVICE (définies dans main.cu) */
__device__ void d_initLocale(memlocale_gpu *ml);
__device__ void d_initTermes(memlocale_gpu *ml);
__device__ void d_initTache(long long numero, memlocale_gpu *ml , char * GrayTab);
__device__ void d_modifierCase(int numCase, memlocale_gpu *ml);
__device__ void d_accumulerTache(memlocale_gpu *ml, char * GrayTab);
__device__ void d_compterTache(long long t, memlocale_gpu *ml, char * GrayTab);

#endif
