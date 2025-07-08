#ifndef ACTVATIONFNH
#define ACTVATIONFNH

#include "matrix.h"
#include <math.h>

void LayerSigmoid(Mat inp){
  float val;
  for(size_t i = 0; i < inp.r * inp.c; i++){
    val = readMat(inp, 0, i);
    setMat(inp, 0, i, 1.f / (exp(-val) + 1.f));
  }
}

void LayerRectify(Mat inp){
  float val;
  for(size_t i = 0; i < inp.r * inp.c; i++){
    val = readMat(inp, 0, i);
    setMat(inp, 0, i, val <= 0.f ? 0.f : val);
  }
}

void LayerHeaviside(Mat inp){
  float val;
  for(size_t i = 0; i < inp.r * inp.c; i++){
    val = readMat(inp, 0, i);
    setMat(inp, 0, i, val < 0.f ? 0.f : 1.f);
  }
}

void LayerSoftmax(Mat inp){
  float total;
  for(unsigned i = 0; i < inp.r; i++){
    total = 0.f;
    for(unsigned j = 0; j < inp.c; j++)
      total += readMat(inp, i, j);
    for(unsigned j = 0; j < inp.c; j++)
      *viewMat(inp, i, j) /= total;
  }
}

float LossSquared(Mat out, Mat targ){
  float total = 0.f;
  for (size_t i = 0; i < out.r * out.c; i++){
    float diff = out.data[i] - targ.data[i];
    total += diff * diff;
  }
  return total;
}

float LossAbsolute(Mat out, Mat targ){
  float total = 0.f;
  for (size_t i = 0; i < out.r * out.c; i++){
    float diff = out.data[i] - targ.data[i];
    total += diff < 0.f ? -diff : diff;
  }
  return total;
}

float LossCategory(Mat out, Mat targ){
  float total = 0.f;
  for (size_t i = 0; i < out.r * out.c; i++){
    float y = targ.data[i], dy = out.data[i];
    if(dy <= 0.f) dy = 1e-7f;
    total -= y * log(dy);
  }
  return total;
}

float LossBinary(Mat out, Mat targ){
  float total = 0.f;
  for (size_t i = 0; i < out.r * out.c; i++){
    float y = targ.data[i], dy = out.data[i];
    if(dy <= 0.f) dy = 1e-7f;
    if(dy >= 1.f) dy = 1.f - 1e-7f;
    total -= y * log(dy) + (1.f - y) * log(1.f - dy);
  }
  return total;
}

#endif