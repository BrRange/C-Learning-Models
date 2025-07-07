#ifndef ACTVATIONFNH
#define ACTVATIONFNH

#include <math.h>

float ModelSigmoid(float inp, int dx){
  inp = 1.f / (1.f + expf(-inp));
  return dx ? inp * (1 - inp) : inp;
}

float ModelRectify(float inp, int dx){
  return inp <= 0.f ? 0.f : dx ? 1.f : inp;
}

float ModelHeaviside(float inp, int dx){
  return (dx || inp < 0.f) ? 0.f : 1.f;
}

#endif