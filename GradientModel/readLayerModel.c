#include <stdio.h>
#include <stdlib.h>
#define ACTIVATIONFNIMPL
#define LAYERMODELIMPL
#include "LayerModel.h"

void printMat(Mat m){
  for(unsigned i = 0; i < m.r; i++){
    putchar('|');
    for(unsigned j = 0; j < m.c; j++){
      printf("%.2f\t", readMat(m, i, j));
    }
    puts("|");
  }
  putchar(10);
}

LayerModel loadLayerModel(FILE* file){
  LayerModel lm;
  fread(&lm.layerSize, sizeof(size_t), 1, file);
  fread(&lm.loss, sizeof(lm.loss), 1, file);
  lm.layer = malloc(sizeof(Layer) * lm.layerSize);
  for(size_t i = 0; i < lm.layerSize; i++){
    fread(&lm.layer[i].act, sizeof(enum LayerFunc), 1, file);
    fread(&lm.layer[i].weight.r, sizeof(unsigned), 1, file);
    fread(&lm.layer[i].weight.c, sizeof(unsigned), 1, file);
    lm.layer[i].weight.data = malloc(lm.layer[i].weight.r * lm.layer[i].weight.c * sizeof(float));
    fread(lm.layer[i].weight.data, sizeof(float), lm.layer[i].weight.r * lm.layer[i].weight.c, file);
    fread(&lm.layer[i].bias.r, sizeof(unsigned), 1, file);
    fread(&lm.layer[i].bias.c, sizeof(unsigned), 1, file);
    lm.layer[i].bias.data = malloc(lm.layer[i].bias.r * lm.layer[i].bias.c * sizeof(float));
    fread(lm.layer[i].bias.data, sizeof(float), lm.layer[i].bias.r * lm.layer[i].bias.c, file);
  }
  return lm;
}

int main(){
  FILE *f = fopen("Model.bin", "rb");
  LayerModel lm = loadLayerModel(f);
  fclose(f);

  Mat input = newMat(1, lm.layer[0].weight.r), output = {};
  fillMat(input,
    1, 1, 1,
    0, 0, 1,
    0, 1, 0,
    0, 0, 1,
    1, 1, 0
  );

  outputLayerModel(lm, input, &output);
  freeMat(input);
  freeLayerModel(lm);
  
  putchar(' ');
  for(int i = 0; i < 10; i++)
    printf("%i  \t", i);
  putchar(10);

  printMat(output);
  freeMat(output);
}