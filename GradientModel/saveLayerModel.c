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

void saveLayerModel(LayerModel lm, FILE* file){
  fwrite(&lm.layerSize, sizeof(size_t), 1, file);
  fwrite(&lm.loss, sizeof(lm.loss), 1, file);
  for(size_t i = 0; i < lm.layerSize; i++){
    fwrite(&lm.layer[i].act, sizeof(enum LayerFunc), 1, file);
    fwrite(&lm.layer[i].weight.r, sizeof(unsigned), 1, file);
    fwrite(&lm.layer[i].weight.c, sizeof(unsigned), 1, file);
    fwrite(lm.layer[i].weight.data, sizeof(float), lm.layer[i].weight.r * lm.layer[i].weight.c, file);
    fwrite(&lm.layer[i].bias.r, sizeof(unsigned), 1, file);
    fwrite(&lm.layer[i].bias.c, sizeof(unsigned), 1, file);
    fwrite(lm.layer[i].bias.data, sizeof(float), lm.layer[i].bias.r * lm.layer[i].bias.c, file);
  }
}

int main(){
  LayerModel lm = newLayerModel(4ull, 15, 16, 16, 16, 10);
  lm.loss = EnumLossCategory;
  lm.layer[0].act = EnumLayerRectify;
  lm.layer[1].act = EnumLayerRectify;
  lm.layer[2].act = EnumLayerRectify;
  lm.layer[3].act = EnumLayerSoftmax;

  LayerData ld = newLayerData(lm, 10);
  fillLayerData(ld,
    #include "imageData.h"
  );

  Mat input = copyMat(ld.input), output = {};

  for(int i = 1; i <= 5000; i++){
    if(i % 100); else printf("Iteration %i\n", i);
    trainLayerModel(lm, ld, 1e-2f, 1e-2f);
  }
  freeLayerData(ld);

  outputLayerModel(lm, input, &output);
  freeMat(input);
  
  printMat(output);
  freeMat(output);

  FILE *f = fopen("Model.bin", "wb");
  saveLayerModel(lm, f);
  fclose(f);
  freeLayerModel(lm);
}