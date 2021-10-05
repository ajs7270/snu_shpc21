#include <stdio.h>

int main() {
    int x;
    scanf("%d", &x);

    unsigned int temp = 1 << 31;

    for(int i = 0; i<sizeof(int)*8; i++){
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