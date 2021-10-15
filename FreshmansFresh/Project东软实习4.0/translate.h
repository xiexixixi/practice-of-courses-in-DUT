#pragma once
#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include"struct.h"
void translate(struct map *MAP, FILE	*fp);
struct map* GenerateMAP(FILE *fp);
void show(struct map *MAP);