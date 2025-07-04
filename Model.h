#ifndef ModelH
#define ModelH

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

struct Model{
  size_t layerSize;
  Layer *layer;
};
typedef struct Model Model;

struct ModelData{
  Matrix input, output;
};
typedef struct ModelData ModelData;

Model newModel(size_t layers, unsigned inp, ...){
  Model model = {layers, malloc(layers * sizeof(Layer))};
  va_list args;
  va_start(args, inp);
  for(size_t i = 0; i < layers; i++){
    unsigned next = va_arg(args, unsigned);
    model.layer[i] = newLayer(inp, next);
    inp = next;
  }
  va_end(args);
  return model;
}

void outputModel(Model model, Matrix in, Matrix *out){
  Matrix_overwrite(out, Matrix_new(in.r, model.layer[model.layerSize - 1ull].weight.c));
  Matrix subOut = {};
  for(unsigned u = 0; u < in.r; u++){
    Matrix subIn = {1, in.c, Matrix_get(in, u, 0)};
    outputLayer(model.layer[0], subIn, &subOut);
    for (size_t i = 1; i < model.layerSize; i++)
      outputLayer(model.layer[i], subOut, &subOut);
    for(unsigned v = 0; v < subOut.c; v++)
      Matrix_set(*out, u, v, subOut.data[v]);
  }
  Matrix_free(subOut);
}

float costModel(Model model, ModelData modelData){
  float total = 0.0f;
  Matrix aux = {};
  outputModel(model, modelData.input, &aux);
  for (size_t i = 0; i < aux.r * aux.c; i++){
    float diff = aux.data[i] - modelData.output.data[i];
    total += diff * diff;
  }
  Matrix_free(aux);
  return total / aux.r;
}

float weightModelData(ModelData modelData, float w, float b){
  float total = 0.f;
  for(unsigned i = 0; i < modelData.input.r; i++)
  for(unsigned k = 0; k < modelData.input.c; k++)
  for(unsigned j = 0; j < modelData.output.c; j++){
    float in = Matrix_read(modelData.input, i, k);
    total += (w * in + b - Matrix_read(modelData.output, i, j)) * in;
  }
  return total / modelData.input.r;
}

float biasModelData(ModelData modelData, float w, float b){
  float total = 0.f;
  for(unsigned i = 0; i < modelData.input.r; i++)
  for(unsigned k = 0; k < modelData.input.c; k++)
  for(unsigned j = 0; j < modelData.output.c; j++){
    float in = Matrix_read(modelData.input, i, k);
    total += (w * in + b - Matrix_read(modelData.output, i, j));
  }
  return total / modelData.input.r;
}

void trainModel(Model model, ModelData modelData, float rate){
  for (size_t l = 0; l < model.layerSize; ++l) {
    Layer ly = model.layer[l];
    for (size_t i = 0; i < ly.weight.r * ly.weight.c; i++)
    for (size_t j = 0; j < ly.bias.r * ly.bias.c; j++){
      ly.weight.data[i] -= rate * weightModelData(modelData, ly.weight.data[i], ly.bias.data[j]);
      ly.bias.data[j] -= rate * biasModelData(modelData, ly.weight.data[i], ly.bias.data[j]);
    }
    /* for (unsigned j = 0; j < ly.bias.c; j++){
      original = ly.bias.data[j];
      ly.bias.data[j] -= eps;
      dcost = costModel(model, modelData);
      ly.bias.data[j] = original + eps;
      dcost = costModel(model, modelData) - dcost;
      dcost /= 2.f * eps;
      ly.bias.data[j] = original - rate * dcost;
    } */
  }
}

void freeModel(Model model){
  for(size_t i = 0; i < model.layerSize; i++)
    freeLayer(model.layer[i]);
  free(model.layer);
}

ModelData newModelData(Model model, unsigned size){
  ModelData modelData = {
    Matrix_new(size, model.layer[0].weight.r),
    Matrix_new(size, model.layer[model.layerSize - 1].weight.c)
  };
  return modelData;
}

void fillModelData(ModelData modelData, ...){
  unsigned ic = modelData.input.c, oc = modelData.output.c;
  va_list args;
  va_start(args, modelData);
  for (unsigned i = 0; i < modelData.input.r; i++) {
    for (unsigned j = 0; j < ic; j++)
      Matrix_set(modelData.input, i, j, va_arg(args, double));
    for (unsigned j = 0; j < oc; j++)
      Matrix_set(modelData.output, i, j, va_arg(args, double));
  }
  va_end(args);
}

void freeModelData(ModelData modelData){
  Matrix_free(modelData.input);
  Matrix_free(modelData.output);
}

#endif