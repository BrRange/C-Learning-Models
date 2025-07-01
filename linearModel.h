#ifndef NNMH
#define NNMH

#include <stdlib.h>
#include <stdarg.h>

struct LinearModel{
    size_t inputSize;
    float *bias;
    float *weight;
    float (*act)(float);
};
typedef struct LinearModel LinearModel;

struct LinearData{
    size_t dataSize, inputSize;
    float **data;
};
typedef struct LinearData LinearData;

LinearModel newLinearModel(size_t size){
    LinearModel lm = {size, malloc(4ull), malloc(4ull * size), 0};
    *lm.bias = 1.f;
    for(size_t i = 0; i < size; i++)
        lm.weight[i] = 1.f;
    return lm;
}

float costLinearModel(LinearModel lm, LinearData ld){
    float res = 0.f;
    for(size_t i = 0; i < ld.dataSize; i++){
        float partial = 0.f;
        for(size_t j = 0; j < lm.inputSize; j++){
            partial += lm.weight[j] * ld.data[i][j];
        }
        partial += *lm.bias;
        if(lm.act) partial = lm.act(partial);
        partial -= ld.data[i][lm.inputSize];
        res += partial * partial;
    }
    return res / ld.dataSize;
}

void trainLinearModel(LinearModel lm, LinearData ld, float eps, float rate){
    float original, dcost;
    for(size_t i = 0; i < lm.inputSize; i++){
        original = lm.weight[i];
        lm.weight[i] += eps;
        dcost = costLinearModel(lm, ld);
        lm.weight[i] = original;
        dcost -= costLinearModel(lm, ld);
        dcost /= 2.f * eps;
        lm.weight[i] -= rate * dcost;
    }
    original = *lm.bias;
    *lm.bias += eps;
    dcost = costLinearModel(lm, ld);
    *lm.bias = original;
    dcost -= costLinearModel(lm, ld);
    dcost /= eps;
    *lm.bias -= rate * dcost;
}

float outputLinearModel(LinearModel lm, ...){
    float output = 0.f;
    va_list args;
    va_start(args, lm);
    for(size_t i = 0; i < lm.inputSize; i++)
        output += va_arg(args, double) * lm.weight[i];
    va_end(args);
    output += *lm.bias;
    return lm.act ? lm.act(output) : output;
}

void freeLinearModel(LinearModel lm){
    free(lm.bias);
    free(lm.weight);
}

LinearData newLinearData(LinearModel lm, size_t size){
    LinearData ld = {size, lm.inputSize, malloc(8ull * size)};
    for(size_t i = 0; i < size; i++)
      ld.data[i] = malloc(4ull * lm.inputSize + 4ull);
    return ld;
}

void fillLinearData(LinearData ld, ...){
    va_list args;
    va_start(args, ld);
    for(size_t i = 0; i < ld.dataSize; i++)
    for(size_t j = 0; j <= ld.inputSize; j++){
        ld.data[i][j] = va_arg(args, double);
    }
    va_end(args);
}

void freeLinearData(LinearData dt){
    for(size_t i = 0; i < dt.dataSize; i++)
        free(dt.data[i]);
    free(dt.data);
}

#endif