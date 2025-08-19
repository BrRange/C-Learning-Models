#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include "layerModel.h"

#define DIAG M_SQRT1_2
#define TAU (2 * M_PI)

void printMat(Mat m){
  for(u32 i = 0; i < m.r; i++){
    putchar('|');
    for(u32 j = 0; j < m.c; j++)
      printf("%.2f\t", matRead(m, i, j));
    puts("|");
  }
  putchar(10);
}

void printLayerModel(LayerModel *lm){
  for(size_t i = 0; i < lm->layerSize; ++i){
    printf("Layer %zu weight:\n", i);
    printMat(lm->layer[i].weight);
    printf("Layer %zu bias:\n", i);
    printMat(lm->layer[i].bias);
  }
}

void saveLayerModel(LayerModel lm, FILE* file){
  fwrite(&lm.layerSize, sizeof(size_t), 1, file);
  fwrite(&lm.loss, sizeof(lm.loss), 1, file);
  for(size_t i = 0; i < lm.layerSize; i++){
    fwrite(&lm.layer[i].act, sizeof(enum LayerFunc), 1, file);
    fwrite(&lm.layer[i].weight.r, sizeof(u32), 1, file);
    fwrite(&lm.layer[i].weight.c, sizeof(u32), 1, file);
    fwrite(lm.layer[i].weight.data, sizeof(float), lm.layer[i].weight.r * lm.layer[i].weight.c, file);
    fwrite(&lm.layer[i].bias.r, sizeof(u32), 1, file);
    fwrite(&lm.layer[i].bias.c, sizeof(u32), 1, file);
    fwrite(lm.layer[i].bias.data, sizeof(float), lm.layer[i].bias.r * lm.layer[i].bias.c, file);
  }
}

int main(){
  LayerModel lm = newLayerModel(4ull, 15, 16, 16, 16, 36);
  lm.loss = EnumLossCategory;
  lm.layer[0].act = EnumLayerRectify;
  lm.layer[1].act = EnumLayerRectify;
  lm.layer[3].act = EnumLayerSoftmax;
  
  LayerData ld = newLayerData(&lm, 44);
  fillLayerData(&ld,
    #include "imageData.h"
  );

  int iteration = 5000;
  printf("Training 0%%");
  for(int i = 1; i <= iteration; ++i){
    if(i % (iteration / 100)); else printf("\rTraining %i%%", i * 100 / iteration);
    trainLayerModel(&lm, &ld, 1e-3f, 1e-1f);
  }
  putchar(10);
  freeLayerData(&ld);

  printMat(readLayerModelOutput(&lm));

  FILE *f = fopen("Model.bin", "wb");
  saveLayerModel(lm, f);
  fclose(f);
  freeLayerModel(&lm);
}