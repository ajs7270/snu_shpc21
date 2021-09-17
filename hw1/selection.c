#include <stdio.h>

void print_array(int a[], int n) {
  for (int i = 0; i < n; ++i) {
    printf("%d ", a[i]);
  }
  printf("\n");
}

void swap(int* x, int *y){
  int temp = *x;
  *x = *y;
  *y = temp;
}

int main() {
  int n;
  scanf("%d", &n);
  int a[n];
  for (int i = 0; i < n; ++i) {
    scanf("%d", &a[i]);
  }

  for (int i = 0; i < n - 1; ++i) {
    /*
     * 1) Find the minimum element among a[i + 1] ~ a[n - 1]
     * 2) Swap a[i] and the minimum element
     */

    // YOUR CODE HERE (~15 lines)
    int min_index = i;
    for(int j = i; j < n; ++j){
	if (a[min_index] > a[j]){
		min_index = j;
	}
    }
    swap(&a[min_index],&a[i]); 

    print_array(a, n);
  }

  return 0;
}
