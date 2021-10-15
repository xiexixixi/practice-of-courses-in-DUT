#pragma once
#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include"sort.h"
#include"translate.h"
void write(struct map *arr, FILE* fp);/*将数据写入文件函数*/
void linkid_binary_search(struct map *arr[], int length);/*linkid检索（二分法检索）*/
void roadname_search(struct map *arr[], int length);/*路名检索*/
void classnumber_search(struct map *arr[], int length);/*Class番号检索*/
void sideway_search(struct map *arr[], int length);/*岔路口检索*/