#ifndef LAYERMODELH
#define LAYERMODELH

#include "../Matrix/matrix.h"

struct Layer{
  size_t inputSize;
  size_t outputSize;
  float *weight;
  float *bias;
  float (*activation)(float);
};
typedef struct Layer Layer;

struct LayerModel{
  size_t layerSize;
  Layer *layer;
};
typedef struct LayerModel LayerModel;

LayerModel newLayerModel(size_t layers, ...){
  LayerModel lm = {layers, malloc(layers * sizeof(Layer))};
  va_list args;
  va_start(args, layers);

  va_end(args);
}

#endif