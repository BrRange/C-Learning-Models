#define ACTIVATIONFNIMPL
#define LAYERMODELIMPL
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "layerModel.h"

void printMat(Mat m){
  for(unsigned i = 0; i < m.r; i++){
    putchar('|');
    for(unsigned j = 0; j < m.c; j++){
      printf("%.2f\t", m.data[i * m.c + j]);
    }
    puts("|");
  }
}

LayerModel loadLayerModel(FILE *file){
  LayerModel lm;
  fread(&lm.layerSize, sizeof(size_t), 1, file);
  fread(&lm.loss, sizeof(lm.loss), 1, file);
  lm.layer = malloc(sizeof(Layer) * lm.layerSize);
  for(size_t i = 0; i < lm.layerSize; i++){
    fread(&lm.layer[i].act, sizeof(enum LayerFunc), 1, file);
    fread(&lm.layer[i].weight.r, sizeof(unsigned), 1, file);
    fread(&lm.layer[i].weight.c, sizeof(unsigned), 1, file);
    lm.layer[i].weight.data = malloc(sizeof(float) * lm.layer[i].weight.r * lm.layer[i].weight.c);
    fread(lm.layer[i].weight.data, sizeof(float), lm.layer[i].weight.r * lm.layer[i].weight.c, file);
    fread(&lm.layer[i].bias.r, sizeof(unsigned), 1, file);
    fread(&lm.layer[i].bias.c, sizeof(unsigned), 1, file);
    lm.layer[i].bias.data = malloc(sizeof(float) * lm.layer[i].bias.r * lm.layer[i].bias.c);
    fread(lm.layer[i].bias.data, sizeof(float), lm.layer[i].bias.r * lm.layer[i].bias.c, file);
    lm.layer[i].output.r = 0;
    lm.layer[i].output.c = lm.layer[i].weight.c;
    lm.layer[i].output.data = NULL;
  }
  return lm;
}

int main(){
  FILE *f = fopen("Model.bin", "rb");
  if(!f){
    puts("File could not be opened");
    return 1;
  }
  LayerModel lm = loadLayerModel(f);
  fclose(f);

  Mat input = {NULL, 1, 15};
  matAlloc(&input);

  Matrix_expr(input, i + 1 + j);

  
  outputLayerModel(&lm, input);
  
  printMat(readLayerModelOutput(&lm));

  matFree(&input);
  freeLayerModel(&lm);
}