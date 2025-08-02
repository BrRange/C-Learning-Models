#include <stdio.h>
#include <stdlib.h>
#include "LayerModel.h"

void printMat(Mat m){
  for(unsigned i = 0; i < m.r; i++){
    putchar('|');
    for(unsigned j = 0; j < m.c; j++){
      printf("%.2f\t", matRead(m, i, j));
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
    lm.layer[i].weight.data = malloc(sizeof(float) * lm.layer[i].weight.r * lm.layer[i].weight.c);
    fread(lm.layer[i].weight.data, sizeof(float), lm.layer[i].weight.r * lm.layer[i].weight.c, file);
    fread(&lm.layer[i].bias.r, sizeof(unsigned), 1, file);
    fread(&lm.layer[i].bias.c, sizeof(unsigned), 1, file);
    lm.layer[i].bias.data = malloc(sizeof(float) * lm.layer[i].bias.r * lm.layer[i].bias.c);
    fread(lm.layer[i].bias.data, sizeof(float), lm.layer[i].bias.r * lm.layer[i].bias.c, file);
  }
  return lm;
}

int main(){
  FILE *f = fopen("Model.bin", "rb");
  LayerModel lm = loadLayerModel(f);
  fclose(f);

  for(u32 i = 0; i < lm.layerSize; ++i){
    lm.layer[i].output = (Mat){0, lm.layer[i].weight.c, 0};
  }

  float stackInp[15];
  Mat input = {1, lm.layer[0].weight.r, stackInp};
  matFill(input,
    1, 1, 0,
    0, 0, 1,
    0, 1, 0,
    0, 0, 1,
    1, 1, 0
  );

  outputLayerModel(&lm, input);
  
  putchar(' ');
  for(int i = 0; i < 10; i++)
  printf("%i  \t", i);
  putchar(10);
  
  printMat(readLayerModelOutput(&lm));

  freeLayerModel(&lm);
}