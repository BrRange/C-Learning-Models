#include <stdio.h>
#include <time.h>
#include "layerModel.h"
#include "activationFn.h"

void printMat(Matrix m){
  for(unsigned i = 0; i < m.r; i++){
    putchar('|');
    for(unsigned j = 0; j < m.c; j++){
      printf("%.2f\t", Matrix_read(m, i, j));
    }
    puts("|");
  }
  putchar(10);
}

int main(){
  srand(time(0));
  LayerModel lm = newLayerModel(2, 2, 2, 1);
  lm.layer[0].act = ModelSigmoid;
  lm.layer[1].act = ModelSigmoid;

  for(int i = 0; i < lm.layerSize; i++){
    printMat(lm.layer[i].weight);
    printMat(lm.layer[i].bias);
  }

  LayerData ld = newLayerData(lm, 4);
  fillLayerData(ld,
    0.f, 0.f, 0.f,
    0.f, 1.f, 1.f,
    1.f, 0.f, 1.f,
    1.f, 1.f, 0.f
  );

  Matrix temp = {};
  Matrix input[4] = {
    Matrix_new(1, 2),
    Matrix_new(1, 2),
    Matrix_new(1, 2),
    Matrix_new(1, 2)
  };
  Matrix_fill_int(input[0], 0, 0);
  Matrix_fill_int(input[1], 0, 1);
  Matrix_fill_int(input[2], 1, 0);
  Matrix_fill_int(input[3], 1, 1);

  printf("%f\n\n", costLayerModel(lm, ld));
  for(size_t i = 0; i < 10000; i++)
    trainLayerModel(lm, ld, 1e-1f, 1e-1f);
  freeLayerData(ld);
  
  printf("%f\n\n", costLayerModel(lm, ld));

  /* for(int i = 0; i < 4; i++){
    outputLayerModel(lm, input[i], &temp);
    printf("%i %i %.2f\n", i & 1, !!(i & 2), temp.data[0]);
  } */
  for(int i = 0; i < lm.layerSize; i++){
    printMat(lm.layer[i].weight);
    printMat(lm.layer[i].bias);
  }
    
  Matrix_free(temp);
  Matrix_free(input[0]);
  Matrix_free(input[1]);
  Matrix_free(input[2]);
  Matrix_free(input[3]);
  freeLayerModel(lm);
}