#pragma once
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
int  refind(FILE *fp, int whatyouwant, int *longsum);
void renew(int longsum, FILE *fp, int flag, int bruch, int dispclass);
void reNew(FILE *Fp, int whatidyouwant, int flag, int bruch, int dispclass);
void update(FILE *fp);