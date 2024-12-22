Legal Notice
============

 Copyright (C) 2017
 Indian Institute of Technology, Roorkee (www.iitr.ac.in)
 The University of Sheffield (www.sheffield.ac.uk)

 This file is part of Entropy based Termination Criterion for 
 Multi-objective Evolutionary Algorithms

 GNU Lesser General Public License Usage
 This file may be used under the terms of the GNU Lesser General
 Public License version 2.1 as published by the Free Software
 Foundation and appearing in the file LICENSE.LGPL included in the
 packaging of this file.  Please review the following information to
 ensure the GNU Lesser General Public License version 2.1 requirements
 will be met: http://www.gnu.org/licenses/old-licenses/lgpl-2.1.html.


Introduction
============
Thanks for downloading this software. You can find out more information about the algorithm by checking the following papers:
1) D. K. Saxena, Arnab Sinha, J. A. Duro, Q. Zhang, "Entropy-Based Termination Criterion for Multiobjective Evolutionary Algorithms", vol. 20, no. 4, pp. 485-498, 2015
2) J. A. Duro, D. K. Saxena, "Timing the Decision Support for Real-World Many-Objective Optimization Problems", In: Trautmann H. et al. (eds) Evolutionary Multi-Criterion Optimization. EMO 2017. Lecture Notes in Computer Science, vol 10173. Springer, 2017

This software applies a dissimilarity measure to quantify the difference between two population of non-dominated solutions for a given multi-objective optimization problem. Each solution is represented by an objective vector.

Instructions towards use of the Codes
=========================================

Entropy Codes are 2 sets - 

1) For computation of dissimilarity measure - entropy.zip
 
2) For statistical analysis towards termination - entropy_dissimilarity_measure.tar.bz

entropy_dissimilarity_measure.tar.bz contains a script file entropy_script.sh which analyse the sample population of solutions corresponding to DTLZ2 with 3 Objectives given in folder "samples". This script file runs the code present in folder "which_gen_single"
