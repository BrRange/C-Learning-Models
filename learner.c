#include <stdio.h>
#include <time.h>
#include "layerModel.h"
#include "activationFn.h"

int main(){
  srand(time(0)); rand();
  LayerModel lm = newLayerModel(3, 2, 4, 2, 1);
  lm.layer[0].act = ModelSigmoid;
  lm.layer[1].act = ModelSigmoid;


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
  Matrix_fill_int(input[1], 1, 0);
  Matrix_fill_int(input[2], 0, 1);
  Matrix_fill_int(input[3], 1, 1);

  for(size_t i = 0; i < 100000; i++)
    trainLayerModel(lm, ld, 1e-1f, 1e-1f);
  freeLayerData(ld);

  for(int i = 0; i < 4; i++){
    outputLayerModel(lm, input[i], &temp);
    printf("%i %i %.2f\n", i & 1, !!(i & 2), temp.data[0]);
  }

  Matrix_free(temp);
  Matrix_free(input[0]);
  Matrix_free(input[1]);
  Matrix_free(input[2]);
  Matrix_free(input[3]);
  freeLayerModel(lm);
}