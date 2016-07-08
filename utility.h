#pragma once
#include<iostream>
#include <fstream>
#include "mat.h"


typedef unsigned int uint;

namespace l0l1IO
{
	void process_input(float*& X, float*& l1graph_alpha, float*& adjmat, float*& A, float*& AtA, float& S1, uint& n, uint& d, uint &nn, const char *input_matfile_name);
}
