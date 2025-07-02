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
  return ly;
}

void outputLayer(Layer ly, Matrix *inp){
  Matrix_overwrite(inp, Matrix_mul(*inp, ly.weight));
  Matrix_add(*inp, ly.bias);
  if (ly.act) Matrix_applyFn(*inp, ly.act);
}

void freeLayer(Layer ly){
  Matrix_free(&ly.weight);
  Matrix_free(&ly.bias);
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
  for(size_t i = 0; i < layers; i++){
    unsigned next = va_arg(args, unsigned);
    lm.layer[i] = newLayer(inp, next);
    inp = next;
  }
  va_end(args);
  return lm;
}

void outputLayerModel(LayerModel model, Matrix *inp) {
  for (size_t i = 0; i < model.layerSize; i++)
    outputLayer(model.layer[i], inp);
}

float costLayerModel(LayerModel lm, LayerData ld, Matrix *aux){
  float total = 0.0f;
  for (unsigned i = 0; i < ld.input.r; i++) {
    Matrix_overwrite(aux, Matrix_extractRow(ld.input, i));
    outputLayerModel(lm, aux);

    for (unsigned j = 0; j < ld.output.c; j++) {
      float diff = aux->data[j] - Matrix_read(ld.output, i, j);
      total += diff * diff;
    }
  }
  return total / ld.input.r;
}

void trainLayerModel(LayerModel lm, LayerData ld, Matrix *aux, float eps, float rate){
  float original, dcost;

  // Loop over each layer
  for (size_t l = 0; l < lm.layerSize; l++) {
    Layer ly = lm.layer[l];
    Matrix W = ly.weight;
    Matrix B = ly.bias;

    // Compute gradients for weights
    for (unsigned i = 0; i < W.r; i++) {
      for (unsigned j = 0; j < W.c; j++) {
        original = Matrix_read(W, i, j);
        Matrix_set(W, i, j, original + eps);
        dcost = costLayerModel(lm, ld, aux);
        Matrix_set(W, i, j, original);
        dcost -= costLayerModel(lm, ld, aux);
        dcost /= 2.f * eps;
        Matrix_set(W, i, j, original - rate * dcost);
      }
    }

    // Compute gradients for biases
    for (unsigned j = 0; j < B.c; j++) {
      original = Matrix_read(B, 0, j);
      Matrix_set(B, 0, j, original + eps);
      dcost = costLayerModel(lm, ld, aux);
      Matrix_set(B, 0, j, original);
      dcost -= costLayerModel(lm, ld, aux);
      dcost /= 2.f * eps;
      Matrix_set(B, 0, j, original - rate * dcost);
    }
  }
}

void freeLayerModel(LayerModel lm){
  for(size_t i = 0; i < lm.layerSize; i++)
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

void fillLayerData(LayerData ld, ...) {
  unsigned ic = ld.input.c;
  unsigned oc = ld.output.c;
  va_list args;
  va_start(args, ld);
  for (unsigned i = 0; i < ld.input.r; i++) {
    for (unsigned j = 0; j < ic; j++)
      Matrix_set(ld.input, i, j, va_arg(args, double));
    for (unsigned j = 0; j < oc; j++)
      Matrix_set(ld.output, i, j, va_arg(args, double));
  }
  va_end(args);
}

void freeLayerData(LayerData ld){
  Matrix_free(&ld.input);
  Matrix_free(&ld.output);
}

#endif