#ifndef LAYERMODELH
#define LAYERMODELH

#include "matrix.h"

struct Layer{
  Matrix weight, bias;
  float (*act)(float);
};
typedef struct Layer Layer;

Layer newLayer(unsigned inp, unsigned out){
  Layer ly = {Matrix_new(inp, out), Matrix_new(1, out), 0};
  Matrix_randomize(ly.weight);
  Matrix_randomize(ly.bias);
  return ly;
}

void outputLayer(Layer ly, Matrix in, Matrix *out){
  Matrix_overwrite(out, Matrix_mul(in, ly.weight));
  Matrix_add(*out, ly.bias);
  if (ly.act) Matrix_applyFn(*out, ly.act);
}

void freeLayer(Layer ly){
  Matrix_free(ly.weight);
  Matrix_free(ly.bias);
}

struct LayerModel{
  size_t layerSize;
  Layer *layer;
};
typedef struct LayerModel LayerModel;

struct LayerData{
  Matrix input, output;
};
typedef struct LayerData LayerData;

LayerModel newLayerModel(size_t layers, unsigned inp, ...){
  LayerModel lm = {layers, malloc(layers * sizeof(Layer))};
  va_list args;
  va_start(args, inp);
  for(size_t i = 0; i < layers; ++i){
    unsigned next = va_arg(args, unsigned);
    lm.layer[i] = newLayer(inp, next);
    inp = next;
  }
  va_end(args);
  return lm;
}

void outputLayerModel(LayerModel model, Matrix in, Matrix *out){
  outputLayer(model.layer[0], in, out);
  for (size_t i = 1; i < model.layerSize; ++i){
    outputLayer(model.layer[i], *out, out);
  }
}

float costLayerModel(LayerModel lm, LayerData ld){
  float total = 0.0f;
  Matrix aux1 = {}, aux2 = {};
  for (unsigned i = 0; i < ld.input.r; ++i) {
    Matrix_overwrite(&aux1, Matrix_extractRow(ld.input, i));
    outputLayerModel(lm, aux1, &aux2);

    for (unsigned j = 0; j < ld.output.c; ++j){
      float diff = aux2.data[j] - Matrix_read(ld.output, i, j);
      total += diff * diff;
    }
  }
  Matrix_free(aux1);
  Matrix_free(aux2);
  return total / ld.input.r;
}

void trainLayerModel(LayerModel lm, LayerData ld, float eps, float rate){
  float original, dcost;
  for (size_t l = 0; l < lm.layerSize; ++l) {
    Layer ly = lm.layer[l];
    for (size_t i = 0; i < ly.weight.r * ly.weight.c; ++i){
      original = ly.weight.data[i];
      ly.weight.data[i] -= eps;
      dcost = costLayerModel(lm, ld);
      ly.weight.data[i] = original + eps;
      dcost = costLayerModel(lm, ld) - dcost;
      dcost /= 2.f * eps;
      ly.weight.data[i] = original - rate * dcost;
    }
    for (unsigned j = 0; j < ly.bias.c; ++j){
      original = ly.bias.data[j];
      ly.bias.data[j] -= eps;
      dcost = costLayerModel(lm, ld);
      ly.bias.data[j] = original + eps;
      dcost = costLayerModel(lm, ld) - dcost;
      dcost /= 2.f * eps;
      ly.bias.data[j] = original - rate * dcost;
    }
  }
}

void freeLayerModel(LayerModel lm){
  for(size_t i = 0; i < lm.layerSize; ++i)
    freeLayer(lm.layer[i]);
  free(lm.layer);
}

LayerData newLayerData(LayerModel lm, unsigned size){
  LayerData ld = {
    Matrix_new(size, lm.layer[0].weight.r),
    Matrix_new(size, lm.layer[lm.layerSize - 1].weight.c)
  };
  return ld;
}

void fillLayerData(LayerData ld, ...){
  unsigned ic = ld.input.c, oc = ld.output.c;
  va_list args;
  va_start(args, ld);
  for (unsigned i = 0; i < ld.input.r; ++i) {
    for (unsigned j = 0; j < ic; ++j)
      Matrix_set(ld.input, i, j, va_arg(args, double));
    for (unsigned j = 0; j < oc; ++j)
      Matrix_set(ld.output, i, j, va_arg(args, double));
  }
  va_end(args);
}

void freeLayerData(LayerData ld){
  Matrix_free(ld.input);
  Matrix_free(ld.output);
}

#endif