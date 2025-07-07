#include <stdio.h> 
#include <stdlib.h> 
#include "Tokenizer.h"

void printTokenizer(Tokenizer tokenizer){
  for(size_t i = 0; i < tokenizer.token->count; i++){
    unsigned key = tokenizer.token->items[i];
    assert(key < tokenizer.pairMap->count);
    if(tokenizer.pairMap->items[key].a == key){
      printf("%c", key);
    } else{
      printf("[%u]", key);
    }
  }
  printf("\n");
}

int main(int argc, const char **argv){
  if(argc < 3){
    puts("Tokenizer [read|dump] [file]");
    return 1;
  }
  Tokenizer tokenizer = newTokenizer();

  const char *filename = argv[2];
  FILE *fp = fopen(filename, "r");
  if (!fp){
    printf("Error: could not open file %s", filename);
    return 1;
  }
  if(argv[1][0] == 'r'){
    readTokenizer(tokenizer, fp);

    return 0;
  }

  initTokenizer(&tokenizer, fp);
  fclose(fp);
  filename = argc > 2 ? argv[3] : "TokenDump";

  bakeTokenizer(tokenizer);
  
  if(dumpTokenizer(tokenizer, filename) == 0){
    puts("Failed to dump the tokenizer");
  } else{
    puts("Tokenizer dumped");
  }

  freeTokenizer(tokenizer);
}