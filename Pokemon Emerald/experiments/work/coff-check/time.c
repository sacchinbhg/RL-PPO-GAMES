#include <time.h>
int main(void){struct timespec ts;return clock_gettime(CLOCK_MONOTONIC,&ts)!=0;}
