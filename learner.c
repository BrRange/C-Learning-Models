#include <stdio.h>
#include "layerModel.h"
#include "activationFn.h"

int main(){
  LayerModel lm = newLayerModel(2, 2, 2, 1);
  lm.layer[0].act = ModelSigmoid;
  lm.layer[1].act = ModelSigmoid;

  LayerData ld = newLayerData(lm, 4);
  fillLayerData(ld,
    0.f, 0.f, 0.f,
    1.f, 0.f, 1.f,
    0.f, 1.f, 1.f,
    1.f, 1.f, 0.f
  );
  Matrix temp = {};
  Matrix input[4] = {
    Matrix_new(1, 2),
    Matrix_new(1, 2),
    Matrix_new(1, 2),
    Matrix_new(1, 2)
  };
  
  while(costLayerModel(lm, ld, &temp) > 0.2f)
    trainLayerModel(lm, ld, &temp, 1e-0f, 1e-0f);
  printf("%f\n", costLayerModel(lm, ld, &temp));
  printf("\n0 0 %.2f\n0 1 %.2f\n1 0 %.2f\n1 1 %.2f\n",
    outputLayerModel(lm, )
  );

  Matrix_free(&temp);
  freeLayerData(ld);
  freeLayerModel(lm);
}