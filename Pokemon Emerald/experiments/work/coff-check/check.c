#include <stdio.h>
#include <stdint.h>
extern int x,y;
extern int *qq(void);
#define MARK(name, sect) __attribute__((section(sect),used,aligned(1))) unsigned char name[1]
MARK(b0,".data$NBA"); MARK(b1,".data$NBZ"); MARK(d0,".data$NDA"); MARK(d1,".data$NDZ");
int main(void) {printf("%p %p %p %p / %p %p %p\n",b0,&x,qq(),b1,d0,&y,d1);return !(b0<(unsigned char*)&x && b0<(unsigned char*)qq() && b1>(unsigned char*)&x && b1>(unsigned char*)qq() && d0<(unsigned char*)&y && d1>(unsigned char*)&y);}
