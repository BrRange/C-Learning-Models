#ifndef MATRIXH
#define MATRIXH

#include <stdarg.h>
#include <stdlib.h>

#define Matrix_expr(mat, expr)\
for(unsigned i = 0; i < mat.r; i++)\
for(unsigned j = 0; j < mat.r; j++)\
mat.data[i * mat.c + j] = (expr)

struct Matrix{
  unsigned r, c;
  float *data;
};
typedef struct Matrix Matrix;

Matrix Matrix_new(unsigned r, unsigned c){
  if(r && c); else return (Matrix){0, 0, 0};
  Matrix mat = {r, c, 0};
  mat.data = calloc(1ull * r * c, 4);
  return mat;
}

Matrix Matrix_identity(unsigned d){
  if(!d) return (Matrix){0, 0, 0};
  Matrix id = {d, d, 0};
  id.data = malloc(4ull * d * d);
  Matrix_expr(id, i == j);
  return id;
}

void Matrix_fill_float(Matrix mat, ...){
  va_list args;
  va_start(args, mat);
  for(unsigned long long sz = 0; sz < mat.r * mat.c; sz++)
    mat.data[sz] = va_arg(args, double);
  va_end(args);
}

void Matrix_fill_int(Matrix mat, ...){
  va_list args;
  va_start(args, mat);
  for(unsigned long long sz = 0; sz < mat.r * mat.c; sz++)
    mat.data[sz] = va_arg(args, int);
  va_end(args);
}

float Matrix_read(Matrix mat, unsigned r, unsigned c){
  return mat.data[r * mat.c + c];
}
float *Matrix_get(Matrix mat, unsigned r, unsigned c){
  return &mat.data[r * mat.c + c];
}
void Matrix_set(Matrix mat, unsigned r, unsigned c, float val){
  mat.data[r * mat.c + c] = val;
}

void Matrix_randomize(Matrix mat){
  for(unsigned long long i = 0; i < mat.r * mat.c; i++)
    mat.data[i] = (float) rand() / (float) RAND_MAX;
}

void Matrix_add(Matrix dst, Matrix val){
  if(dst.r != val.r || dst.c != val.c) return;
  for(unsigned long long i = 0; i < dst.r * dst.c; i++)
    dst.data[i] += val.data[i];
}

Matrix Matrix_mul(Matrix a, Matrix b){
  if(a.c != b.r) return (Matrix){0, 0, 0};
  Matrix mat = Matrix_new(a.r, b.c);
  for(unsigned i = 0; i < a.r; i++)
    for(unsigned k = 0; k < a.c; k++)
      for(unsigned j = 0; j < b.c; j++)
        *Matrix_get(mat, i, j) += Matrix_read(a, i, k) * Matrix_read(b, k, j);
  return mat;
}

void Matrix_applyFn(Matrix m, float (*fn)(float)){
  for(unsigned long long i = 0; i < m.r * m.c; i++)
    m.data[i] = fn(m.data[i]);
}

Matrix Matrix_extractRow(Matrix m, unsigned r){
  Matrix ret = Matrix_new(1, m.c);
  float *row = Matrix_get(m, r, 0);
  for(unsigned i = 0; i < m.c; i++)
    ret.data[i] = row[i];
  return ret;
}

void Matrix_destroy(Matrix *mat){
  free(mat->data);
  *mat = (Matrix){0, 0, 0};
}

void Matrix_free(Matrix mat){
  free(mat.data);
}

void Matrix_overwrite(Matrix *var, Matrix val){
  free(var->data);
  *var = val;
}

Matrix Matrix_transpose(Matrix mat){
  Matrix ret = Matrix_new(mat.c, mat.r);
  for (unsigned i = 0; i < mat.r; i++)
    for (unsigned j = 0; j < mat.c; j++)
      *Matrix_get(ret, j, i) = Matrix_read(mat, i, j);
  return ret;
}

Matrix Matrix_reduce(Matrix mat, unsigned x, unsigned y){
  if(mat.r && mat.c); else return (Matrix){0, 0, 0};
  Matrix res = Matrix_new(mat.r - 1, mat.c - 1);
  size_t count = 0;
  for(unsigned i = 0; i < mat.r; i++){
    if(i == y) continue;
    for(unsigned j = 0; j < mat.c; j++){
      if(j == x) continue;
      res.data[count] = Matrix_read(mat, i, j);
      count++;
    }
  }
  return res;
}

float Matrix_determinant(Matrix mat){
  if(mat.r != mat.c) return (0.f / 0.f);
  if(mat.r && mat.c); else return (0.f / 0.f);
  if(mat.r == 1){
    return mat.data[0];
  }
  else if(mat.r == 2){
    return mat.data[0] * mat.data[3] - mat.data[1] * mat.data[2];
  }
  else if(mat.r == 3){
    return mat.data[0] * (mat.data[4] * mat.data[8] - mat.data[5] * mat.data[7])
    + mat.data[1] * (mat.data[5] * mat.data[6] - mat.data[3] * mat.data[8])
    + mat.data[2] * (mat.data[3] * mat.data[7] - mat.data[4] * mat.data[6]);
  }
  else{
    float coSum = 0.f;
    for(unsigned i = 0; i < mat.r; i++){
      Matrix temp = Matrix_reduce(mat, i, 0);
      coSum += mat.data[i] * (i & 1 ? -1.f : 1.f) * Matrix_determinant(temp);
      free(temp.data);
    }
    return coSum;
  }
}

#endif