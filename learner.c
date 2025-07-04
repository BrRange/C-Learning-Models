#include <stdio.h>
#include <time.h>
#include "layerModel.h"
#include "activationFn.h"

int main(){
  srand(time(0)); rand();
  LayerModel lm = newLayerModel(2, 2, 2, 1);
  lm.layer[0].act = ModelSigmoid;


  LayerData ld = newLayerData(lm, 4);
  fillLayerData(ld,
    0.f, 0.f, 0.f,
    1.f, 0.f, 1.f,
    0.f, 1.f, 1.f,
    1.f, 1.f, 0.f
  );

  for(size_t i = 0; i < 10000; i++)
    trainLayerModel(lm, ld, 1e-1f, 1e-1f);
  freeLayerData(ld);

  Matrix input = Matrix_new(4, 2);
  Matrix_fill_int(input,
    0, 0,
    1, 0,
    0, 1,
    1, 1
  );
  Matrix output = {};

  outputLayerModel(lm, input, &output);
  freeLayerModel(lm);
  Matrix_free(input);
  
  for(int i = 0; i < 4; i++)
    printf("%i %i %.2f\n", i & 1, !!(i & 2), Matrix_read(output, i, 0));
  
  Matrix_free(output);
}