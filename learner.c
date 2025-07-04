#include <stdio.h>
#include <time.h>
#include "Model.h"
#include "activationFn.h"

int main(){
  srand(time(0)); rand();
  Model lm = newModel(2, 2, 2, 1);
  lm.layer[0].act = ModelSigmoid;

  ModelData ld = newModelData(lm, 4);
  fillModelData(ld,
    0.f, 0.f, 0.f,
    1.f, 0.f, 1.f,
    0.f, 1.f, 1.f,
    1.f, 1.f, 0.f
  );

  Matrix input = Matrix_new(4, 2);
  Matrix_fill_int(input,
    0, 0,
    1, 0,
    0, 1,
    1, 1
  );
  Matrix output = {};

  printf("Cost before: %f\n", costModel(lm, ld));
  
  for(size_t i = 0; i < 10000; i++)
    trainModel(lm, ld, 1e-1f);

  printf("Cost after: %f\n", costModel(lm, ld));
  freeModelData(ld);

  outputModel(lm, input, &output);
  freeModel(lm);
  Matrix_free(input);

  for(int i = 0; i < 4; i++)
    printf("%i %i -> %.2f\n", i & 1, i / 2, output.data[i]);

  Matrix_free(output);
}