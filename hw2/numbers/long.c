#include <stdio.h>

int main() {
  long x;
  scanf("%ld", &x);

    unsigned long temp = 1;
    temp <<= sizeof(long)*8 - 1;

    for(int i = 0; i<sizeof(long)*8; i++){
        if(temp & x){
            printf("%d",1);
        }else{
            printf("%d",0);
        }
        temp >>= 1;
    }
    printf("\n");

  return 0;
}
