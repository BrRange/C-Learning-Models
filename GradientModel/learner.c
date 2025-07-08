#include <stdio.h> 
#include <stdlib.h> 
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

int main(int argc, const char **argv){
  LayerModel lm = newLayerModel(2ull, 2, 2, 1);
  lm.loss = LossBinary;
  lm.layer[0].act = LayerSigmoid;

  LayerData ld = newLayerData(lm, 4);
  fillLayerData(ld,
    0.f, 0.f, 0.f,
    1.f, 0.f, 1.f,
    0.f, 1.f, 1.f,
    1.f, 1.f, 0.f
  );

  Mat input = copyMat(ld.input), output = {};

  for(int i = 0; i < 10000; i++) trainLayerModel(lm, ld, 1e-2f, 1e-2f);
  freeLayerData(ld);

  outputLayerModel(lm, input, &output);
  freeLayerModel(lm);
  freeMat(input);

  printMat(output);
  freeMat(output);
}