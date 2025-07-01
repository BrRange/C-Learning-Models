#ifndef ACTVATIONFNH
#define ACTVATIONFNH

#include <math.h>

float ModelSigmoid(float inp){
  float ex = expf(inp);
  return ex / (1.f + ex);
}

float ModelRectify(float inp){
  return inp <= 0.f ? 0.f : inp;
}

float ModelHeavyside(float inp){
  return inp >= 0.f ? 1.f : 0.f;
}

#endif