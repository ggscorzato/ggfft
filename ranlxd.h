/*******************************************************************************
*
* file ranlxd.h
*
* Copyright (C) 2005 Martin Luescher
*
* This software is distributed under the terms of the GNU General Public
* License (GPL)
*
*******************************************************************************
*   modified by M. Cristoforetti including gaussian generator
*******************************************************************************/

#ifndef _RANLXD_H
#define _RANLXD_H

void ranlxd(double r[],int n);
void rlxd_init(int level,int seed);
int rlxd_size(void);
void rlxd_get(int state[]);
void rlxd_reset(int state[]);
/* generate a gaussian vector with <y^2>=1 */
void gauss_vectord(double v[],int n);
void testGauss();
double ranxd();
#endif
