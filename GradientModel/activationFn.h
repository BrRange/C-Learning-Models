#ifndef ACTVATIONFNH
#define ACTVATIONFNH

#include "lowmat.h"
#include <math.h>

void LayerSigmoid(Mat inp){
  float val;
  u32 ite = inp.r * inp.c;
  for(u32 i = 0; i < ite; i++){
    val = inp.data[i];
    inp.data[i] = 1.f / (exp(-val) + 1.f);
  }
}

void LayerRectify(Mat inp){
  float val;
  u32 ite = inp.r * inp.c;
  for(u32 i = 0; i < ite; i++){
    val = inp.data[i];
    inp.data[i] = val <= 0.f ? 0.f : val;
  }
}

void LayerSoftmax(Mat inp){
  float total;
  for(u32 i = 0; i < inp.r; i++){
    total = 0.f;
    for(u32 j = 0; j < inp.c; j++)
      total += expf(matRead(inp, i, j));
    for(u32 j = 0; j < inp.c; j++)
      matWrite(inp, i, j, expf(matRead(inp, i, j)) / total);
  }
}

enum LayerFunc{
  EnumLayerLinear,
  EnumLayerSigmoid,
  EnumLayerRectify,
  EnumLayerSoftmax
};

void (*LayerFuncList[])(Mat) = {
  NULL,
  LayerSigmoid,
  LayerRectify,
  LayerSoftmax
};

float LossSquared(Mat out, Mat targ){
  float total = 0.f;
  u32 ite = out.r * out.c;
  for (u32 i = 0; i < ite; i++){
    float diff = out.data[i] - targ.data[i];
    total += diff * diff;
  }
  return total;
}

float LossAbsolute(Mat out, Mat targ){
  float total = 0.f;
  u32 ite = out.r * out.c;
  for (u32 i = 0; i < ite; i++){
    float diff = out.data[i] - targ.data[i];
    total += diff < 0.f ? -diff : diff;
  }
  return total;
}

float LossCategory(Mat out, Mat targ){
  float total = 0.f;
  u32 ite = out.r * out.c;
  for (u32 i = 0; i < ite; i++){
    float z = out.data[i];
    if(z <= 0.f) z = 1e-7f;
    total += targ.data[i] * log(z);
  }
  return -total;
}

float LossBinary(Mat out, Mat targ){
  float total = 0.f;
  for (u32 i = 0; i < out.r * out.c; i++){
    float y = targ.data[i], z = out.data[i];
    if(z <= 0.f) z = 1e-7f;
    if(z >= 1.f) z = 1.f - 1e-7f;
    total -= y * log(z) + (1.f - y) * log(1.f - z);
  }
  return total;
}

enum LossFunc{
  EnumLossSquared,
  EnumLossAbsolute,
  EnumLossCategory,
  EnumLossBinary
};

float (*LossFuncList[])(Mat, Mat) = {
  LossSquared,
  LossAbsolute,
  LossCategory,
  LossBinary
};

#endif