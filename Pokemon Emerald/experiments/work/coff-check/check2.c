#include <stdio.h>
#include <stdint.h>
extern int x,y;
extern int *qq(void);
#define MARK(name, sect) __attribute__((section(sect),used,aligned(1))) unsigned char name[1]
extern unsigned char b0[], b1[]; __asm__(".section .bss$NBA,\"bw\"\n.globl b0\nb0: .space 1\n.section .bss$NBZ,\"bw\"\n.globl b1\nb1: .space 1\n.text"); MARK(d0,".data$NDA"); MARK(d1,".data$NDZ");
int main(void) {printf("%p %p %p %p / %p %p %p\n",b0,&x,qq(),b1,d0,&y,d1);return !(b0<(unsigned char*)&x && b0<(unsigned char*)qq() && b1>(unsigned char*)&x && b1>(unsigned char*)qq() && d0<(unsigned char*)&y && d1>(unsigned char*)&y);}
