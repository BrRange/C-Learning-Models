#ifndef ModelH
#define ModelH

#include "matrix.h"

struct Layer{
  Mat weight, bias;
  float (*act)(float, int);
};
typedef struct Layer Layer;

Layer newLayer(unsigned inp, unsigned out){
  Layer ly = {newMat(inp, out), newMat(1, out), 0};
  randMat(ly.weight);
  randMat(ly.bias);
  return ly;
}

void outputLayer(Layer ly, Mat in, Mat *out){
  overwriteMat(out, composeMat(in, ly.weight));
  addMat(*out, ly.bias);
  if (ly.act) applyFnMat(*out, ly.act);
}

void freeLayer(Layer ly){
  freeMat(ly.weight);
  freeMat(ly.bias);
}

struct Model{
  size_t layerSize;
  Layer *layer;
};
typedef struct Model Model;

struct ModelData{
  Mat input, output;
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

void outputModel(Model model, Mat in, Mat *out){
  overwriteMat(out, newMat(in.r, model.layer[model.layerSize - 1ull].weight.c));
  Mat subOut = {};
  for(unsigned u = 0; u < in.r; u++){
    Mat subIn = {1, in.c, viewMat(in, u, 0)};
    outputLayer(model.layer[0], subIn, &subOut);
    for (size_t i = 1; i < model.layerSize; i++)
      outputLayer(model.layer[i], subOut, &subOut);
    for(unsigned v = 0; v < subOut.c; v++)
      setMat(*out, u, v, subOut.data[v]);
  }
  freeMat(subOut);
}

float costModel(Model model, ModelData data){
  float total = 0.0f;
  Mat aux = {};
  outputModel(model, data.input, &aux);
  for (size_t i = 0; i < aux.r * aux.c; i++){
    float diff = aux.data[i] - data.output.data[i];
    total += diff * diff;
  }
  freeMat(aux);
  return total / aux.r;
}

float weightModelData(ModelData data, float w, float b){
  float total = 0.f;
  for(unsigned i = 0; i < data.input.r; i++)
  for(unsigned k = 0; k < data.input.c; k++)
  for(unsigned j = 0; j < data.output.c; j++){
    float in = readMat(data.input, i, k);
    total += (w * in + b - readMat(data.output, i, j)) * in;
  }
  return total / data.input.r;
}

float biasModelData(ModelData data, float w, float b){
  float total = 0.f;
  for(unsigned i = 0; i < data.input.r; i++)
  for(unsigned k = 0; k < data.input.c; k++)
  for(unsigned j = 0; j < data.output.c; j++){
    float in = readMat(data.input, i, k);
    total += (w * in + b - readMat(data.output, i, j));
  }
  return total / data.input.r;
}

void trainModel(Model model, ModelData data, float rate) {
  Mat rowOut = {};
  for(unsigned i = 0; i < data.input.r; i++){
    Mat view = {1, data.input.c, viewMat(data.input, i, 0)};
    outputModel(model, view, &rowOut);
    for(unsigned j = 0; j < data.output.c; j++){
      
    }
  }
}

void freeModel(Model model){
  for(size_t i = 0; i < model.layerSize; i++)
    freeLayer(model.layer[i]);
  free(model.layer);
}

ModelData newModelData(Model model, unsigned size){
  ModelData data = {
    newMat(size, model.layer[0].weight.r),
    newMat(size, model.layer[model.layerSize - 1].weight.c)
  };
  return data;
}

void fillModelData(ModelData data, ...){
  unsigned ic = data.input.c, oc = data.output.c;
  va_list args;
  va_start(args, data);
  for (unsigned i = 0; i < data.input.r; i++) {
    for (unsigned j = 0; j < ic; j++)
      setMat(data.input, i, j, va_arg(args, double));
    for (unsigned j = 0; j < oc; j++)
      setMat(data.output, i, j, va_arg(args, double));
  }
  va_end(args);
}

void freeModelData(ModelData data){
  freeMat(data.input);
  freeMat(data.output);
}

#endif