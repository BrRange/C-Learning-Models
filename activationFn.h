#ifndef ACTVATIONFNH
#define ACTVATIONFNH

#include <math.h>

float ModelSigmoid(float inp){
  return 1.f / (1.f + expf(-inp));
}

float ModelRectify(float inp){
  return inp <= 0.f ? 0.f : inp;
}

float ModelHeavyside(float inp){
  return inp >= 0.f ? 1.f : 0.f;
}

#endif